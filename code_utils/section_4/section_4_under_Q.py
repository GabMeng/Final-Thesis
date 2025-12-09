import numpy as np
import math
from scipy.special import gamma as Gamma
from scipy.integrate import quad
from scipy.optimize import differential_evolution
from scipy.optimize import minimize


def causal_conv_fft(dW2, a):
    """
    dW2: (M_eff, max_steps)
    a:   (max_steps,)
    returns X: (M_eff, max_steps) with causal convolution along axis=1
    """
    M_eff, T = dW2.shape
    n_conv = 1 << ((2*T - 1).bit_length())  # next pow2 ≥ 2T-1
    F_dW2 = np.fft.rfft(dW2, n=n_conv, axis=1)
    F_a   = np.fft.rfft(a[None, :], n=n_conv, axis=1)  # broadcast once
    Xfull = np.fft.irfft(F_dW2 * F_a, n=n_conv, axis=1)[:, :T]
    return Xfull


def build_pricer_state(cfg: dict, alpha: float) -> dict:
    """
    Prepare fixed, reusable simulation state for Volterra MC pricing.

    Expected cfg keys:
      - M: int                     # base number of MC paths
      - n: int                     # time steps per year
      - T_list: List[float]        # maturities in years
      - S0: float
      - r: float                   # flat risk-free (cc)
      - q: float                   # flat dividend (cc)
      - y0: float | None           # initial log-spot; if None -> log(S0)
      - antithetic: bool           # if True, stack sign-flipped normals
      - seed: int | None           # RNG seed for common random numbers

    Returns a dict with:
      - alpha, dt, sqrt_dt
      - steps_by_T: Dict[T, int]
      - nr_steps: List[int]           (aligned with T_list)
      - max_steps: int
      - Z1, Z2: np.ndarray            (standard normals, shape (M_eff, max_steps))
      - M_eff: int
      - disc_by_T: Dict[T, float]
      - y0, r, q, S0, n, M, T_list
    """
    # Pull config with sensible defaults
    M        = int(cfg["M"])
    n        = int(cfg["n"])
    T_list   = list(cfg["T_list"])
    S0       = float(cfg["S0"])
    r        = float(cfg["r"])
    q        = float(cfg["q"])
    y0       = math.log(float(cfg["S0"]))
    antithetic = bool(cfg.get("antithetic", True))
    seed     = cfg.get("seed", 42)

    # Time grid stats
    dt = 1.0 / n
    sqrt_dt = math.sqrt(dt)

    # Steps per maturity (at least 1 step if T > 0)
    steps_by_T = {}
    nr_steps = []
    for T in T_list:
        steps = int(round(n * float(T)))
        if T > 0.0 and steps < 1:
            steps = 1
        steps_by_T[T] = steps
        nr_steps.append(steps)
    max_steps = max(nr_steps) if nr_steps else 0
    if max_steps <= 0:
        raise ValueError("max_steps is zero — provide positive maturities in T_list.")

    # Common random numbers
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    Z1 = rng.standard_normal(size=(M, max_steps))
    Z2 = rng.standard_normal(size=(M, max_steps))

    # Antithetic variates (variance reduction)
    if antithetic:
        Z1 = np.vstack([Z1, -Z1])
        Z2 = np.vstack([Z2, -Z2])

    M_eff = Z1.shape[0]

    # Discount factors per maturity (risk-neutral)
    disc_by_T = {T: math.exp(-r * float(T)) for T in T_list}

    return {
        "alpha": alpha,
        "dt": dt,
        "sqrt_dt": sqrt_dt,
        "steps_by_T": steps_by_T,
        "nr_steps": nr_steps,
        "max_steps": max_steps,
        "Z1": Z1,
        "Z2": Z2,
        "M_eff": M_eff,
        "disc_by_T": disc_by_T,
        "y0": y0,
        "r": r,
        "q": q,
        "S0": S0,
        "n": n,
        "M": M,
        "T_list": T_list,
        "antithetic": antithetic,
        "seed": seed,
    }

def kernel_comte_renault(x, g, alpha, k):
    if x <= 0.0:
        return 0.0

    def integrand(u, k_, alpha_):
        return math.exp(k_ * u) * (u ** alpha_)

    int_arg, _ = quad(integrand, 0.0, x, args=(k, alpha), limit=50)
    r = (g / Gamma(1.0 + alpha)) * (x**alpha - k * math.exp(-k * x) * int_arg)
    return r


def build_kernel_vector(max_steps, dt, g, alpha, k):
    xs = np.arange(1, max_steps + 1) * dt
    # cumulative Simpson increments for I(x)
    I = np.zeros_like(xs)
    if max_steps > 0:
        f_prev = 0.0  # f(0) = 0^alpha * e^{0} = 0 for alpha > -1
        acc = 0.0
        for j, x in enumerate(xs, start=1):
            u0 = (j-1)*dt; u1 = x; um = 0.5*(u0+u1)
            f0 = f_prev
            fm = math.exp(k*um) * (um**alpha)
            f1 = math.exp(k*u1) * (u1**alpha)
            acc += (dt/6.0) * (f0 + 4.0*fm + f1)
            I[j-1] = acc
            f_prev = f1
    # r(x) = g/Gamma(1+alpha) * ( x^alpha - k e^{-k x} I(x) )
    factor = g / Gamma(1.0 + alpha)
    a = factor * (xs**alpha - k * np.exp(-k*xs) * I)
    return a



def price_grid(params, state, strikes_by_T):
    rho, g, k, ln_sigma0 = map(float, params)

    # pull
    alpha   = state["alpha"]; dt = state["dt"]; sqrt_dt = state["sqrt_dt"]
    Z1      = state["Z1"];    Z2 = state["Z2"]; M_eff   = state["M_eff"]
    steps_by_T = state["steps_by_T"]; max_steps = state["max_steps"]
    disc_by_T  = state["disc_by_T"];  y0 = state["y0"]
    r = state["r"]; q = state["q"]; T_list = state["T_list"]

    # correlated Brownian increments
    W1  = Z1
    W2i = rho * Z1 + np.sqrt(max(1.0 - rho*rho, 1e-12)) * Z2
    dW1 = sqrt_dt * W1[:, :max_steps]
    dW2 = sqrt_dt * W2i[:, :max_steps]

    # kernel vector up to max_steps
    a_vec = build_kernel_vector(max_steps, dt, g, alpha, k)

    # Volterra convolution for ALL steps at once
    X_full = causal_conv_fft(dW2, a_vec)       # (M_eff, max_steps)

    # log-vol path 
    ln_sigma_full = ln_sigma0 + np.c_[np.zeros((M_eff, 1)), X_full]
    sigma_full = np.exp(ln_sigma_full)

    # log-price increments for the whole horizon
    incr = (r - q) * dt + sigma_full[:, :-1] * dW1 - 0.5 * (sigma_full[:, :-1]**2) * dt

    # cumulative log-price 
    Y = np.empty((M_eff, max_steps + 1), float)
    Y[:, 0] = y0
    np.cumsum(incr, axis=1, out=Y[:, 1:])
    Y[:, 1:] += y0
    S = np.exp(Y)

    # price per maturity by slicing
    prices = {}
    for T in T_list:
        steps = steps_by_T[T]
        ST = S[:, steps]  # terminal at this maturity
        Ks = np.asarray(strikes_by_T[T], float)
        payoff = np.maximum(ST[:, None] - Ks[None, :], 0.0)
        prices[T] = disc_by_T[T] * payoff.mean(axis=0)

    return prices



def vega_weighted_mse(model_prices, market_prices, vegas):
    num = 0.0
    den = 0.0
    for T in market_prices:
        w = np.asarray(vegas[T])
        w = np.maximum(w, 1e-8)  # floor to avoid division by zero
        diff = np.asarray(model_prices[T]) - np.asarray(market_prices[T])
        num += np.sum(w * (diff ** 2))
        den += np.sum(w)
    return num / max(den, 1e-12)

def vega_weighted_mse_2(model_prices, market_prices, vegas, vega_power=0.3):
    """
    Hybrid weighting: combines vega weighting with uniform weighting
    wing_weight: 0.0 = pure vega weighting, 1.0 = uniform weighting
    """
    num = 0.0
    den = 0.0
    for T in market_prices:
        vega_raw = np.asarray(vegas[T])
        vega_raw = np.maximum(vega_raw, 1e-8)

        # Normalize vega to have mean = 1
        vega_norm = vega_raw / np.mean(vega_raw)

        #uniform weights
        w = (1 - vega_power) * vega_norm + vega_power * np.ones_like(vega_norm)

        diff = np.asarray(model_prices[T]) - np.asarray(market_prices[T])
        num += np.sum(w * (diff ** 2))
        den += np.sum(w)
    return num / max(den, 1e-12)

def butterfly_penalty_mat(C):  # C: shape (nT, nK), common K grid
    # second-difference along strikes >= 0
    B = C[:, :-2] - 2.0*C[:, 1:-1] + C[:, 2:]
    return np.square(B[B < 0.0]).sum()

def slope_penalty_mat(C, K):
    dK = np.diff(K)[None, :]                  # (1, nK-1)
    dC = np.diff(C, axis=1) / dK              # (nT, nK-1)
    p = 0.0
    p += np.square(dC[dC > 0.0]).sum()        # must be <= 0
    p += np.square((dC[dC < -1.0] + 1.0)).sum()  # must be >= -1
    return p

def calendar_penalty_mat(C):  # C: shape (nT, nK) in increasing T
    D = C[:-1, :] - C[1:, :]     # want C_{t2} >= C_{t1} → violations are positive
    V = D[D > 0.0]
    return np.square(V).sum()

def price_scale(C):
    # median ATM price across maturities as scale
    return np.median(C[:, C.shape[1]//2])


def make_objective(state, market_prices, vegas, strikes_by_T,
                   verbose=False, penalty=1e6,
                   w_bfly=1e-3, w_slope=1e-3, w_cal=1e-3, normalize_penalties=True, vega_power= 0.2):
    """
    Modified to calibrate ln_sigma0 as well - removed from parameters.
    """
    T_order = sorted(state["T_list"])
    K = np.asarray(strikes_by_T[T_order[0]], float)
    nT = len(T_order); nK = K.size


    def objective(theta):
        try:
            th = np.asarray(theta, float).ravel()
            rho, g, k, ln_sigma0 = th  
            params = (rho, g, k, ln_sigma0)

            # model prices as a matrix (nT, nK)
            grids = price_grid(params, state, strikes_by_T)
            C_mod = np.vstack([np.asarray(grids[T], float) for T in T_order])

            # main fit: vega-weighted MSE in PRICE space
            #val = np.sum(W_vega * (C_mod - C_mkt)**2) / np.sum(W_vega)
            val = vega_weighted_mse_2(grids, market_prices, vegas, vega_power)

            # penalties
            p_bfly = butterfly_penalty_mat(C_mod)
            p_slope = slope_penalty_mat(C_mod, K)
            p_cal = calendar_penalty_mat(C_mod)

            if normalize_penalties:
                sc = price_scale(C_mod)
                sc2 = (sc*sc if sc > 1e-12 else 1.0)
                p_bfly /= sc2
                p_slope /= sc2
                p_cal   /= sc2

            return float(val + w_bfly*p_bfly + w_slope*p_slope + w_cal*p_cal)
        except Exception:
            return float(penalty)

    return objective



def calibrate(params0, bounds, state,
              strikes_by_T, market_prices, vegas,
              method="de", maxiter=60, popsize=15,
              show_progress=True, polish=True, workers=1,
              fast_mode=False, w_bfly=0.0, w_slope=0.0, w_cal=0.0, normalize_penalties=True, vega_power=0.2):
    """
    Calibrate (rho, g, k, ln_sigma0).
    Enhanced with better progress tracking for DE.
    """
    # optionally thin paths (keep time grid intact) for speed
    if fast_mode:
        state = dict(state)  # shallow copy
        Z1 = state["Z1"]; Z2 = state["Z2"]
        half = max(1, Z1.shape[0] // 2)
        state["Z1"] = Z1[:half]
        state["Z2"] = Z2[:half]
        state["M_eff"] = half

    obj = make_objective(state, market_prices, vegas, strikes_by_T,
                         verbose=False, penalty=1e6,
                         w_bfly=w_bfly, w_slope=w_slope, w_cal=w_cal,
                         normalize_penalties=normalize_penalties, vega_power=vega_power)

    if method == "de":
        # Enhanced progress tracking
        import time
        start_time = time.time()

        progress_info = {
            "generation": 0,
            "best_obj": float('inf'),
            "best_params": None,
            "start_time": start_time,
            "eval_count": 0
        }

        def enhanced_callback(xk, convergence=None):
            if show_progress:
                progress_info["generation"] += 1
                current_obj = obj(xk)  # Evaluate current best
                progress_info["eval_count"] += 1

                # Update if we found a better solution
                if current_obj < progress_info["best_obj"]:
                    progress_info["best_obj"] = current_obj
                    progress_info["best_params"] = xk.copy()
                    improvement = "✓ NEW BEST"
                else:
                    improvement = ""

                # Time tracking
                elapsed = time.time() - progress_info["start_time"]
                eta = (elapsed / progress_info["generation"]) * (maxiter - progress_info["generation"]) if progress_info["generation"] > 0 else 0

                #
                rho, g, k, ln_sigma0 = xk[:4]

                print(f"Gen {progress_info['generation']:2d}/{maxiter} | "
                      f"Obj: {current_obj:.6f} | "
                      f"ρ={rho:+.4f} g={g:.4f} k={k:.4f} ln_σ₀={ln_sigma0:.4f} | "
                      f"Time: {elapsed:.1f}s ETA: {eta:.1f}s {improvement}")

                if convergence is not None:
                    print(f"         Convergence: {convergence:.8f}")

            return False 

        print(f"Starting Differential Evolution:")
        print(f"Population size: {popsize}, Max generations: {maxiter}")
        print(f"Bounds: ρ∈{bounds[0]}, g∈{bounds[1]}, k∈{bounds[2]}, ln_σ₀∈{bounds[3]}")
        print(f'Params: M : {state["M"]}, n : {state["n"]}, rf : {state["r"]}')
        print(f'Maturities : {state["T_list"]}')
        print("=" * 80)

        result = differential_evolution(
            obj,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            mutation = (0.8, 1.9),  #questo [0,2)
            recombination = 0.8, #questo [0,1]
            callback=enhanced_callback,
            polish=polish,
            updating="deferred",
            init="sobol",
            workers=workers,
            tol=1e-4,
            disp=False  
        )

        theta_star = np.asarray(result.x, float)
        best_val = float(result.fun)

        # Final summary
        elapsed_total = time.time() - start_time
        print("=" * 80)
        print(f"DE COMPLETED in {elapsed_total:.1f}s")
        print(f"Final objective: {best_val:.8f}")
        print(f"Total function evaluations: {result.nfev}")
        print(f"Success: {result.success}")
        if hasattr(result, 'message'):
            print(f"Message: {result.message}")
        print(f"Termination reason: {result.message}")
        print(f"Number of iterations: {result.nit}")
        print(f"Success flag: {result.success}")

    else:
        # Local optimization with simpler progress
        import time
        start_time = time.time()
        evals = {"n": 0, "start_time": time.time()}

        def wrapped(th):
            val = obj(th)
            if show_progress:
                evals["n"] += 1
                if evals["n"] % 10 == 0:
                    elapsed = time.time() - evals["start_time"]
                    rho, g, k, ln_sigma0 = th[:4]  
                    print(f"{method} eval {evals['n']:3d}: obj={val:.6f} | "
                          f"ρ={rho:+.4f} g={g:.4f} k={k:.4f} ln_σ₀={ln_sigma0:.4f} | {elapsed:.1f}s")
            return val

        result = minimize(
            wrapped,
            x0=np.asarray(params0, float),
            method=method,
            bounds=bounds,
            options=dict(maxiter=maxiter)
        )
        theta_star = np.asarray(result.x, float)
        best_val = float(result.fun)

    best_params = {"rho": float(theta_star[0]),
                   "g":   float(theta_star[1]),
                   "k":   float(theta_star[2]),
                   "ln_sigma0": float(theta_star[3])} 

    return best_params, best_val


