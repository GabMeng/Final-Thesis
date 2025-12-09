import pandas as pd
import numpy as np
from scipy.special import gamma, erf
import math
from scipy.integrate import quad
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import differential_evolution
from scipy import optimize
from scipy.optimize import minimize
import random as rd
from scipy.stats import norm
from scipy.optimize import brentq
from tqdm import tqdm
from math import log, sqrt, exp, isfinite



def Gamma(x):
    return gamma(x)
    

def kernel_comte_renault(x, g, alpha, k):
    def integrand(u, k, alpha):
        return math.exp(k*u)*u**alpha
    
    int_arg, _ = quad(integrand, 0, x, args=(k, alpha))
    r = (g/Gamma(1+alpha))*(x**alpha - k*math.exp(-k*x)*int_arg)
    return r 

def simulate_paths_basic_MC(M : int, n : int, T : float, y_0, ln_sigma_0, rho, k, g, alpha, r, q):
    '''
    Args: 
        M (int) : number of MC paths
        n (int) : number of timesteps, must be normalized to 1 ie, 1/n is the nr of time steps in 1 year
        T (int) : time to maturity.
        
        
    '''
    
    dt = 1/n
    
    nr_of_steps = int(round(n * T))
    
    sqrt_dt = np.sqrt(dt)
    
    W1 = np.random.normal(size=(M, nr_of_steps))
    W2 = np.random.normal(size=(M, nr_of_steps))
    
    W2 = rho*W1 + np.sqrt(1-rho**2)*W2
    
    dW1 = sqrt_dt*W1
    dW2 = sqrt_dt*W2
    
    
    print("Calculating Volterra conv.")
    a_vec = np.array([kernel_comte_renault(j*dt, g, alpha, k) for j in range(1, nr_of_steps+1)]) # kernell computation 
    print("Volterra conv. done!")
    
    X = np.empty((M, nr_of_steps))
    for i in tqdm(range(nr_of_steps), desc="Calculating Volatility Path"):
        X[:, i] = (dW2[:, :i+1] * a_vec[i::-1]).sum(axis=1) # volterra convolution 
    

       
        
    ln_sigma = ln_sigma_0 + np.c_[np.zeros((M, 1)), X]
    sigma = np.exp(ln_sigma)

    
    
    incr = (r - q) * dt + sigma[:, :-1] * dW1 - 0.5 * (sigma[:, :-1]**2) * dt

    Y = np.empty((M, nr_of_steps + 1))
    Y[:, 0] = y_0
    Y[:, 1:] = y_0 + np.cumsum(incr, axis=1)

    
    S = np.exp(Y)

    
    Y_mean = Y.mean(axis=0)
    mc_error = 3 * S.std(axis=0, ddof=1) / np.sqrt(M)
    vol_mean = sigma.mean(axis=0)
    S_mean = S.mean(axis=0)
    
    
    
    return Y_mean, vol_mean, mc_error, S_mean, S




def interp_risk_free(option_chain, risk_free_curve):
    
    curve_rf = risk_free_curve[["maturity", "df"]].sort_values("maturity").to_numpy()
    maturity_rf = curve_rf[:, 0]
    mid_rf = curve_rf[:, 1]
    log_mid_rd = np.log(mid_rf)
    
    
    # converting the conventions, rf is /360 and opt is /252
    maturity_opt = option_chain["time_to_maturity"] #.to_numpy()*(252/360)
    
    idx_upper = np.searchsorted(maturity_rf, maturity_opt, side="left")
    
    #boolean mask "out of bound" index
    before_first = idx_upper == 0
    after_last = idx_upper == len(curve_rf)
    
    i_2 = np.clip(idx_upper, 1, len(maturity_rf)-1)
    i_1 = i_2 - 1
    
    T_1, T_2 = maturity_rf[i_1], maturity_rf[i_2]
    weight = (maturity_opt - T_1) / (T_2 - T_1)
    log_P = log_mid_rd[i_1] + weight*(log_mid_rd[i_2] - log_mid_rd[i_1])
    
    exact = (idx_upper < len(maturity_rf)) & (maturity_rf[idx_upper] == maturity_opt)
    log_P[exact] = log_mid_rd[idx_upper[exact]]
    
    
    log_P[before_first] = log_mid_rd[0]
    log_P[after_last] = log_mid_rd[-1]
    
    out = option_chain.copy()
    out["discount_rate"] = np.exp(log_P)
    
    return out

def preprocess_option_chain(df, price_today, date_ref, risk_free_curve, fwd_type = "empirical", rf = 0.03):
    """
    Preprocess the option chain DataFrame:
    - Computes time to maturity in years (252 days/year)
    - Sorts by time to maturity
    - Drops rows with time_to_maturity > 12
    - Drops rows where price or ivol are NaN
    - Adds discount_rate, fwd, and moneyness columns

    Args:
        df (pd.DataFrame): Option chain DataFrame with 'maturity' and 'strike'
        price_today (float): Current spot price
        date_ref (pd.Timestamp): Reference date for time to maturity
        risk_free_curve (pd.DataFrame): Risk-free curve for discounting

    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    df = df.copy()
    df["time_to_maturity"] = ((df["maturity"] - date_ref).dt.days) / 252
    df = df.sort_values("time_to_maturity")
    df = df[df["time_to_maturity"] <= 12]
    # Drop rows where price or ivol are NaN
    df = df.dropna(subset=["price", "ivol"])
    if fwd_type == "empirical":
        df = interp_risk_free(df, risk_free_curve)
        df["fwd"] = price_today / df["discount_rate"]
    elif fwd_type == "theoretical":
        df["fwd"] = price_today*np.exp(rf*df["time_to_maturity"])
        df["discount_rate"] = np.exp(-rf *df["time_to_maturity"])
    df["moneyness"] = df["strike"] / df["fwd"]
    return df


def svi_slice(df : pd.DataFrame, ttm, mon_col = "moneyness", time_col = "time_to_maturity", ivol_col = "ivol"):
    '''
    The function fit the best parameters of a SVI fucntion as defined in Gatheral 2004, note that this optimizer works for a slice of the surface, ie the same
    time to maturity.
    
    Args:
        - df : the dataframe with the option chain
        - ttm : the slice we want to fit
        - mon_col : the column where the moneyness is defined m = K/fwd
        - time_col : the time to maturity column (in years)
        - ivol_col : the column of implied volatility
    '''
    
    df_to_svi = pd.DataFrame({
        "log_moneyness" : np.log(df[mon_col]),
        "ttm" : df[time_col],
        "ivol" : df[ivol_col]
        })
    
    df_to_svi = df_to_svi[df_to_svi["ttm"] == ttm].copy()
    df_to_svi["tot_ivar"] = np.pow(df_to_svi["ivol"], 2)*df_to_svi["ttm"]
    
    k = df_to_svi["log_moneyness"].to_numpy(dtype = float)
    w_obs = df_to_svi["tot_ivar"].to_numpy(dtype=float)
    
    def obj_sp(x):
        '''
        We have, make sure the index corresponds: x = [a, b, rho, m, sigma]
        '''
        real_tot_var = w_obs
        total_var_svi = x[0] + x[1]*(x[2]*(k - x[3]) + np.sqrt(np.pow((k - x[3]), 2) + np.pow(x[4], 2)))
        loss_function = np.sum(np.pow((total_var_svi - real_tot_var), 2))
        if np.any(np.isinf(loss_function)):
            print("one value is inf")
            return 1e50
        return loss_function
    return {"data" : df_to_svi, "k" : k, "w_obs" : w_obs, "loss": obj_sp}


def svi_total_variance(params, k):
    """Gatheral SVI total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    a, b, rho, m, sigma = params
    diff = k - m
    return a + b * (rho * diff + np.sqrt(diff**2 + sigma**2))


def evaluate_slice(params, k_grid, ttm):
    """Evaluate IV(k,T) from SVI params (w -> iv)."""
    w_model = svi_total_variance(params, k_grid)
    iv_model = np.sqrt(np.maximum(w_model, 0.0) / ttm)
    return iv_model

def calibration_svi_slice_constrained(
    df, ttm, prev_params=None, mon_col="moneyness", time_col="time_to_maturity", ivol_col="ivol",
    constraint_ks=None, constraint_density=60, eps_calendar=0.0, soft_penalty_lambda=0.0,
    plot=False, strike_grid_dimension=100, maxiter=1000
):
    """
    Optimize params [a,b,rho,m,sigma] with inequality constraints:
        for all k in constraint grid: w_i(k; x) - w_{i-1}(k; prev_params) >= eps_calendar
    If prev_params is None, falls back to unconstrained fit for the first slice.
    Optionally adds a soft penalty on violations to help feasibility.
    """
    arg = svi_slice(df, ttm, mon_col, time_col, ivol_col)
    data = arg["data"]
    k_obs = arg["k"]
    w_obs = arg["w_obs"]

    # Initial guess
    a0 = 0.5 * (np.min(w_obs))
    b0 = 0.1
    rho0 = -0.5
    m0 = 0.0
    sigma0 = 0.1
    x0 = np.array([a0, b0, rho0, m0, sigma0], dtype=float)

    #Bounds
    eps = 1e-8
    bounds = [(eps, max(w_obs)), (eps, 1 - eps), (-1 + eps, 1 - eps), (2 * np.min(k_obs), 2 * np.max(k_obs)), (eps, 1)]

    # Constraint grid in k
    if constraint_ks is None:
        k_lo, k_hi = float(np.min(k_obs)), float(np.max(k_obs))
        # include observed ks + fill with a uniform grid to guard between quotes
        k_fill = np.linspace(k_lo, k_hi, constraint_density)
        constraint_ks = np.unique(np.concatenate([k_obs, k_fill]))
    else:
        constraint_ks = np.asarray(constraint_ks, dtype=float)

    # Previous slice total variance function
    if prev_params is not None:
        def w_prev(kv):
            return svi_total_variance(prev_params, kv)
    else:
        w_prev = None

    # Base (data) objective on w
    def base_obj(x):
        w_model = svi_total_variance(x, k_obs)
        return np.sum((w_model - w_obs) ** 2)

    # Optional soft penalty to help (kept zero by default)
    def penalty_term(x):
        if w_prev is None or soft_penalty_lambda <= 0.0:
            return 0.0
        w_i = svi_total_variance(x, constraint_ks)
        viol = np.maximum((w_prev(constraint_ks) + eps_calendar) - w_i, 0.0)
        return soft_penalty_lambda * np.sum(viol ** 2)

    def obj_with_penalty(x):
        val = base_obj(x) + penalty_term(x)
        if not np.isfinite(val):
            return 1e50
        return val

    # Hard inequality constraints for SLSQP
    constraints = []
    if w_prev is not None:
        # Build one scalar constraint per k in the grid: g(x) = w_i(k;x) - w_prev(k) - eps >= 0
        for kk in constraint_ks:
            def g_factory(k_fixed):
                return lambda x: svi_total_variance(x, k_fixed) - (w_prev(k_fixed) + eps_calendar) #non linear part of the problem we recover w that has a square root in the parametrization
            constraints.append({"type": "ineq", "fun": g_factory(float(kk))})

    # If there's no previous slice, we just do unconstrained
    method = "SLSQP" if (w_prev is not None) else "L-BFGS-B"
    res = minimize(
        obj_with_penalty, x0, method=method, bounds=bounds,
        constraints=constraints if w_prev is not None else (),
        options=dict(maxiter=maxiter, ftol=1e-12, disp=False)
    )

    if plot:
        params = res.x
        k_data = data["log_moneyness"].to_numpy()
        iv_data = data[ivol_col].to_numpy()
        k_grid = np.linspace(min(k_data), max(k_data), strike_grid_dimension)
        iv_model = evaluate_slice(params, k_grid, ttm)
        plt.figure(figsize=(7, 4))
        plt.scatter(k_data, iv_data, label="Observed IV", alpha=0.8)
        plt.plot(k_grid, iv_model, label="SVI (constrained)" if w_prev is not None else "SVI", lw=2)
        plt.xlabel("log-moneyness ln(K/F)")
        plt.ylabel("Implied Volatility")
        plt.title(f"SVI fit at T={ttm}")
        plt.legend(); plt.grid(alpha=0.3); plt.show()

        # Quick visual check of w_i >= w_{i-1} on constraint grid
        if w_prev is not None:
            w_i = svi_total_variance(params, constraint_ks)
            w_p = w_prev(constraint_ks)
            plt.figure(figsize=(7, 4))
            plt.plot(constraint_ks, w_i, label="w_i(k)")
            plt.plot(constraint_ks, w_p + eps_calendar, label="w_{i-1}(k)+eps", linestyle="--")
            plt.xlabel("log-moneyness ln(K/F)")
            plt.ylabel("Total variance w(k)")
            plt.title("Calendar constraint check on grid")
            plt.legend(); plt.grid(alpha=0.3); plt.show()

    return {"fitted_params": res.x, "message": res.message, "success": res.success, "result": res}

def svi_calibration_multislice_constrained(
    df: pd.DataFrame,
    min_ttm, max_ttm,
    number_of_mat=None,
    mon_col="moneyness", time_col="time_to_maturity", ivol_col="ivol",
    plotting_smile=False, k_dimension=100,
    plotting_surface=True, ks_number=100, manual_ks_grid=None,
    eps_calendar=0.0,         # tiny >0 
    constraint_density=80,    # how dense the calendar-constraint k-grid is
    soft_penalty_lambda=0.0   # >0 to help feasibility if needed
):
    """
    Sequentially fits SVI slices T_1 < T_2 < ... using SLSQP with calendar constraints:
        w(k, T_i) >= w(k, T_{i-1}) + eps_calendar  for all k in the constraint grid.
    """
    df_to_svi = pd.DataFrame({
        "log_moneyness": np.log(df[mon_col]),
        "ttm": df[time_col],
        "ivol": df[ivol_col],
    })

    df_to_svi = df_to_svi[(df_to_svi["ttm"] >= min_ttm) & (df_to_svi["ttm"] <= max_ttm)].copy()
    df_to_svi["tot_ivar"] = np.power(df_to_svi["ivol"], 2) * df_to_svi["ttm"]

    if number_of_mat is not None:
        sort_mat = list(df_to_svi["ttm"].value_counts().index)
        if len(sort_mat) < number_of_mat:
            print("The amount of maturities selected is more than the one present; keeping them all...")
            mat_to_keep = sort_mat
        else:
            print(f"Maturities in range: {len(sort_mat)}; keeping the first {number_of_mat} most traded.")
            mat_to_keep = sort_mat[:number_of_mat]
        df_to_svi = df_to_svi[df_to_svi["ttm"].isin(mat_to_keep)]

    unique_mat = sorted(list(df_to_svi["ttm"].unique()))
    if len(unique_mat) == 0:
        raise ValueError("No maturities in the requested range.")

    # Common k-grid for plotting/evaluation (based on the shortest T slice support)
    k_lo, k_hi = (
        np.min(df_to_svi.loc[df_to_svi["ttm"] == unique_mat[0], "log_moneyness"]),
        np.max(df_to_svi.loc[df_to_svi["ttm"] == unique_mat[0], "log_moneyness"]),
    )
    ks_grid = np.linspace(k_lo, k_hi, ks_number) if manual_ks_grid is None else np.asarray(manual_ks_grid, dtype=float)

    results = {}
    strikes = {}
    ivol_surface = np.zeros((len(unique_mat), len(ks_grid)))
    ivol_surf_dict = {}

    prev_params = None
    for idx, t in enumerate(unique_mat):
        # choose a constraint grid for this slice (cover plotting grid + observed points)
        k_obs = df_to_svi.loc[df_to_svi["ttm"] == t, "log_moneyness"].to_numpy(float)
        k_lo_t, k_hi_t = float(np.min(k_obs)), float(np.max(k_obs))
        k_dense = np.linspace(k_lo_t, k_hi_t, constraint_density)
        constraint_ks = np.unique(np.concatenate([k_dense, ks_grid, k_obs]))

        out = calibration_svi_slice_constrained(
            df, t,
            prev_params=prev_params,
            mon_col=mon_col, time_col=time_col, ivol_col=ivol_col,
            constraint_ks=constraint_ks,
            constraint_density=constraint_density,
            eps_calendar=eps_calendar,
            soft_penalty_lambda=soft_penalty_lambda,
            plot=plotting_smile,
            strike_grid_dimension=k_dimension
        )
        params_t = out["fitted_params"]
        results[t] = params_t
        strikes[t] = [k_lo_t, k_hi_t]

        # fill surface row
        ivol_surface[idx, :] = evaluate_slice(params_t, ks_grid, t)
        ivol_surf_dict[t] = ivol_surface[idx, :].copy()

        # advance previous
        prev_params = params_t

    if plotting_surface:
        maturities_plot = np.asarray(unique_mat)
        strikes_plot = np.asarray(ks_grid)
        iv_plot = np.asarray(ivol_surface)
        S, T = np.meshgrid(strikes_plot, maturities_plot)
        iv_masked = np.ma.masked_invalid(iv_plot)

        # IV surface
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(S, T, iv_masked, linewidth=0, antialiased=True, cmap="viridis")
        ax.set_xlabel("Log-Moneyness")
        ax.set_ylabel("Time to maturity (years)")
        ax.set_zlabel("Implied vol")
        ax.set_title("SVI IV Surface (calendar-constrained)")
        fig.colorbar(surf, shrink=0.6, aspect=12, label="Implied vol")
        plt.tight_layout(); plt.show()

        # Total variance surface
        iv_squared = np.power(iv_masked, 2)
        tot_ivar_to_plot = iv_squared * maturities_plot[:, None]
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(S, T, tot_ivar_to_plot, linewidth=0, antialiased=True, cmap="viridis")
        ax.set_xlabel("Log-Moneyness")
        ax.set_ylabel("Time to maturity (years)")
        ax.set_zlabel("Total Implied Variance")
        ax.set_title("Total Variance Surface (monotone in T by construction)")
        fig.colorbar(surf, shrink=0.6, aspect=12, label="Implied var")
        plt.tight_layout(); plt.show()

    return results, strikes, ks_grid, ivol_surface, ivol_surf_dict, unique_mat


def call_price_vega_from_ivs(mat : list, df : pd.DataFrame, ln_moneyness : np.array, ivs_matrix : np.array):
    J, K = ivs_matrix.shape
    calls_price = np.empty(shape=(J, K))
    vegas = np.empty((J, K))
    
    phi_cdf = lambda z : 0.5*(1.0 + erf(z/np.sqrt(2)))
    phi_pdf = lambda z: np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    
    price_dict = {}
    vega_dict = {}
    
    for j, t in enumerate(mat):
        row_t = df[df["time_to_maturity"] == t].iloc[0, :]
        D_t = row_t["discount_rate"]
        FWD_t = row_t["fwd"]
        
        x = ln_moneyness
        sigma = ivs_matrix[j, :]
        
        w = np.pow(sigma, 2)*t
        sqrt_w = np.sqrt(np.maximum(w, 1e-12))
        
        d1 = (-x +0.5*w)/sqrt_w
        d2 = d1-sqrt_w
        
        K_strike = FWD_t*np.exp(x)
        
        calls_price[j, :] = D_t*(FWD_t*phi_cdf(d1) - K_strike*phi_cdf(d2))
        vegas[j, :] = D_t * FWD_t * phi_pdf(d1) * np.sqrt(t)
        
        price_dict[t] = calls_price[j,:]
        vega_dict[t] = vegas[j, :]
        
    return calls_price, vegas, price_dict, vega_dict




def option_pricer(df, maturities: list, log_money_strikes: dict, M: int, n: int, 
                       S0: float, ln_sigma0: float, rho: float, k: float, g: float, alpha: float, 
                       r: float, q: float, type = "call"):
    
    print(f"Simulating the {type} option prices with the params set: rho = {rho}, k : {k}, g : {g}.")
    print("\n")
    print(f"Maturity Range: {min(list(maturities))} to {max(list(maturities))}. Total {len(list(maturities))} maturities.")
    print("\n")
    print(f"Log-Moneyness range : {min(list(log_money_strikes))} to {max(list(log_money_strikes))}. With {len(list(log_money_strikes))} points.")
    print("="*25)
    
    mat_list = list(maturities)
    mat_list = sorted(mat_list)
    T = max(mat_list)
    y_0 = np.log(S0) 
    df = df.copy()
    
    option_prices = {}
    
    print(f"Calculating price path via MC, shape: {(M, n*T)}")
    _, _, _, _, S_path = simulate_paths_basic_MC(M, n, T, y_0, ln_sigma0, rho, k, g, alpha, r, q)
    print("Simulated paths calculated!")
    print("="*25)
    # find the time step that closely matches the maturities and price 
    max_steps = int(round(n*T))
    
    for m in mat_list:
        
        print(f"Pricing for maturity: {m}")
        
        #get back the normal strike to price 
        row_mat = df[df["time_to_maturity"] == m].iloc[0, :]
        fwd_mat = row_mat["fwd"]
        normal_strike = np.exp(log_money_strikes)*fwd_mat
        
        
        closest_step = int(round(n*m))
        closest_step = min(closest_step, max_steps)
        S_m = S_path[:, closest_step]
        discount = np.exp(-r*m)
        
        if type.lower() == "call":
            payoff = np.maximum(S_m[:, None] - normal_strike[None, :], 0.0)
            mean_payoff = payoff.mean(axis = 0)
            opt_price = discount*mean_payoff
        if type.lower() == "put":
            payoff = np.maximum(-S_m[:, None] + normal_strike[None, :], 0.0)
            mean_payoff = payoff.mean(axis = 0)
            opt_price = discount*mean_payoff
            
        option_prices[m] = opt_price
    print("="*25)
    print("Pricing done!")
    print("="*25)
    
    return option_prices

##################################################################################
def bs_price(St, K, r, T, sbs, opt_type = "call"):
    d_1 = ((np.log(St/K)) + (r+((sbs**2)/2))*T) / (sbs*np.sqrt(T))
    d_2 = d_1 - sbs*np.sqrt(T)
    if opt_type.lower() == "call":
        return norm.cdf(d_1)*St - norm.cdf(d_2)*K*np.exp(-r*T)
    elif opt_type.lower() == "put":
        return K*np.exp(-r*T)*norm.cdf(-d_2) - St*norm.cdf(-d_1)
    
def ivol_from_model(S0, K, T, mkt_price, r, tol=1e-4, opt_type="call"):
    lo, hi = 0.0, 5.0
    while True:
        mid = 0.5 * (lo + hi)
        price = bs_price(S0, K, r, T, mid, opt_type)
        if abs(price - mkt_price) < tol:
            return mid
        if price > mkt_price:
            hi = mid
        else:
            lo = mid

##################################################################################
        
def black76_call(F, K, T, sigma, D):
    """
    Black-76 call on forwards. Returns discounted price.
    Handles edge cases robustly.
    """
    # intrinsic if no time or no vol
    if T <= 0.0 or sigma <= 0.0:
        return D * max(F - K, 0.0)

    v = sigma * sqrt(T)
    # guard against ill inputs
    if not (isfinite(F) and isfinite(K) and F > 0.0 and K > 0.0 and isfinite(v)):
        return np.nan

    # d1, d2 (use v directly for better conditioning)
    lf = log(F / K)
    d1 = (lf + 0.5 * v * v) / v
    d2 = d1 - v

    
    return D * (F * norm.cdf(d1) - K * norm.cdf(d2))


# --- implied vol solver (robust) ---
def black76_iv(price, F, K, T, D, tol=1e-10, max_iter=100):
    """
    Solve Black-76 implied vol for a *discounted* call price.
    Robust near expiry and for extreme strikes.
    """
    # basic sanity
    if not all(map(np.isfinite, [price, F, K, T, D])) or F <= 0.0 or K <= 0.0 or D <= 0.0:
        return np.nan

    # no-arbitrage bounds for discounted call
    p_min = D * max(F - K, 0.0)
    p_max = D * F  # loose but safe upper bound

    # outside bounds -> impossible
    if price < p_min - 1e-12 or price > p_max + 1e-12:
        return np.nan

    # if T = 0, return tiny positive vol if strictly above intrinsic, else NaN/0
    if T <= 0.0:
        return 0.0 if price <= p_min + 1e-14 else np.nan

    
    tv = price - p_min
    if tv <= 0.0:
        # scale the epsilon with DF*F and sqrt(T): it disappears as T->0 but is big enough to bracket
        eps_abs = 1e-14 * D * F
        eps_sqrtT = 1e-8 * D * F * sqrt(max(T, 0.0))
        price = p_min + max(eps_abs, eps_sqrtT)
        tv = price - p_min

    # objective
    def f(sig):
        return black76_call(F, K, T, sig, D) - price

    # Initial bracket
    lo = 1e-8
    hi = 3.0
    flo = f(lo)
    fhi = f(hi)

    
    grow = 2.0
    it = 0
    while (np.sign(flo) == np.sign(fhi)) and it < 30 and hi < 1000.0:
        hi *= grow
        fhi = f(hi)
        it += 1

    
    it2 = 0
    while (np.sign(flo) == np.sign(fhi)) and it2 < 10 and lo > 1e-12:
        lo *= 0.5
        flo = f(lo)
        it2 += 1

    # If still failed to bracket, return NaN 
    if np.sign(flo) == np.sign(fhi):
        return np.nan

    # Try a few damped Newton steps inside [lo, hi], then Brent as fallback
    # (Newton is fast when vega is decent; we clamp to bracket to ensure convergence.)
    def vega(sig):
        v = sig * sqrt(T)
        if v <= 0.0:
            return 0.0
        lf = log(F / K)
        d1 = (lf + 0.5 * v * v) / v
        return D * F * sqrt(T) * (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * d1 * d1)

    # Start near-the-money with Brenner-Subrahmanyam-ish guess (crude but safe):
    
    sigma0 = max(lo, min(hi, (price / (D * F)) * np.sqrt(2.0 * np.pi) / max(sqrt(T), 1e-12)))
    sig = sigma0
    for _ in range(8):
        val = f(sig)
        veg = vega(sig)
        if veg <= 0.0 or not np.isfinite(veg):
            break
        step = val / veg
        # damp the step in the tails
        step = np.clip(step, -0.5 * (hi - lo), 0.5 * (hi - lo))
        sig_new = sig - step
        # keep inside the bracket
        sig = float(np.clip(sig_new, lo, hi))
        # Update bracket using sign
        vval = f(sig)
        if vval == 0.0:
            return sig
        if np.sign(vval) == np.sign(flo):
            lo, flo = sig, vval
        else:
            hi, fhi = sig, vval

    # Brent for the guaranteed finish
    try:
        return brentq(f, lo, hi, xtol=tol, maxiter=max_iter)
    except Exception:
        return np.nan


# --- Build IV surface from prices on a k=ln(K/F_T) grid ---

def implied_vol_surface_from_logmoneyness(price_grid, logm_grid, S, r, q=0.0, maturity_min=None, maturity_max=None):
    """
    price_grid: dict[T] -> array of **discounted** call prices C(T,K) on a common k-grid
    logm_grid: array of k values used for all maturities, where k = ln(K/F_T)
    S: spot, r: risk-free cc rate, q: dividend cc yield (default 0)
    maturity_min: minimum maturity to include (optional)
    maturity_max: maximum maturity to include (optional)
    """
    iv_surface = {}
    for T, prices in price_grid.items():
        # Filter by maturity range if specified
        if maturity_min is not None and T < maturity_min:
            continue
        if maturity_max is not None and T > maturity_max:
            continue
            
        F = S * exp((r - q) * T)        # forward
        D = exp(-r * T)                 # discount factor
        Ks = F * np.exp(logm_grid)      # K = F * e^k  
        ivs = np.empty_like(prices, dtype=float)
        for i, (K, p) in enumerate(zip(Ks, prices)):
            ivs[i] = black76_iv(p, F, K, T, D)
        iv_surface[T] = ivs
    return iv_surface

def ivs_compare(ivs_model, ivs_market, maturity_min=None, maturity_max=None):
    """
    Compare model vs market implied vol surfaces using MAE (basis points).

    Parameters
    ----------
    ivs_model : dict[float, np.ndarray]
    ivs_market : dict[float, np.ndarray]
        Dicts keyed by time to maturity with arrays on same moneyness grid.
    maturity_min : float, optional
        Minimum maturity to include in comparison
    maturity_max : float, optional
        Maximum maturity to include in comparison

    Returns
    -------
    diffs : dict[float, np.ndarray]
        Per-maturity absolute errors in basis points.
    """
    # Filter maturities based on range
    filtered_model = {}
    filtered_market = {}
    
    for mat in ivs_model:
        if maturity_min is not None and mat < maturity_min:
            continue
        if maturity_max is not None and mat > maturity_max:
            continue
        if mat in ivs_market:  # Only include if both model and market have this maturity
            filtered_model[mat] = ivs_model[mat]
            filtered_market[mat] = ivs_market[mat]
    
    # shape 
    n_mat_mkt, n_k_mkt = len(filtered_market), len(list(filtered_market.values())[0]) if filtered_market else 0
    n_mat_mod, n_k_mod = len(filtered_model), len(list(filtered_model.values())[0]) if filtered_model else 0
    print(f"Market IVs: {n_mat_mkt} maturities × {n_k_mkt} k-points")
    print(f"Model  IVs: {n_mat_mod} maturities × {n_k_mod} k-points")

    if n_mat_mkt != n_mat_mod or n_k_mkt != n_k_mod:
        print("Shape mismatch — check input surfaces.\n")

    # mat mae
    diffs = {}

    for mat in filtered_model:
        arr_model = np.asarray(filtered_model[mat], dtype=float)
        arr_mkt = np.asarray(filtered_market[mat], dtype=float)
        diff = arr_model - arr_mkt

        abs_diff_bps = np.abs(diff) * 1e4
        diffs[mat] = abs_diff_bps

    return diffs

def ivs_bucket_mae(ivs_model, ivs_market, kgrid, maturity_min=None, maturity_max=None):
    """
    Compute and print MAE (in bps) between model and market IVs
    in ATM, mid, and wings buckets for each maturity and overall.

    Parameters
    ----------
    ivs_model : dict[float, np.ndarray]
        Model implied vols keyed by time to maturity.
    ivs_market : dict[float, np.ndarray]
        Market implied vols keyed by same maturities and grid.
    kgrid : array-like
        Common log-forward moneyness grid.
    maturity_min : float, optional
        Minimum maturity to include in analysis
    maturity_max : float, optional
        Maximum maturity to include in analysis

    Returns
    -------
    per_maturity : dict[T, dict[str, float]]
        {'atm', 'mid', 'wings', 'overall'} MAE (in bps) per maturity.
    overall : dict[str, float]
        Same metrics aggregated across all maturities.
    """
    # Filter maturities based on range
    filtered_model = {}
    filtered_market = {}
    
    for T in ivs_model.keys():
        if maturity_min is not None and T < maturity_min:
            continue
        if maturity_max is not None and T > maturity_max:
            continue
        if T in ivs_market:  # Only include if both model and market have this maturity
            filtered_model[T] = ivs_model[T]
            filtered_market[T] = ivs_market[T]
    
    k = np.asarray(kgrid, dtype=float)
    abs_k = np.abs(k)

    # Define buckets
    buckets = {
        "ATM"   : abs_k <= 0.05,
        "MID"   : (abs_k > 0.05) & (abs_k <= 0.15),
        "WINGS" : (abs_k > 0.15) & (abs_k <= 0.50)
    }

    per_maturity = {}
    all_diffs = {b: [] for b in list(buckets.keys()) + ["OVERALL"]}

    print("\n=== MAE BY MONEYNESS BUCKET (in bps) ===")
    print("Maturity |   ATM   |   MID   |  WINGS  | Overall")

    for T in sorted(filtered_model.keys()):
        mod = np.asarray(filtered_model[T], dtype=float)
        mkt = np.asarray(filtered_market[T], dtype=float)
        diff_bps = np.abs(mod - mkt) * 1e4  # convert to bps

        stats_T = {}
        for name, mask in buckets.items():
            stats_T[name] = float(np.mean(diff_bps[mask])) if np.any(mask) else np.nan
            if np.any(mask):
                all_diffs[name].append(diff_bps[mask])

        stats_T["OVERALL"] = float(np.mean(diff_bps))
        all_diffs["OVERALL"].append(diff_bps)

        per_maturity[T] = stats_T

        print(f"{T:8.2f} | {stats_T['ATM']:7.3f} | {stats_T['MID']:7.3f} | "
              f"{stats_T['WINGS']:7.3f} | {stats_T['OVERALL']:8.3f}")

    # Overall aggregation across all maturities
    overall = {
        name: float(np.mean(np.concatenate(all_diffs[name]))) if len(all_diffs[name]) else np.nan
        for name in all_diffs
    }

    print("=" * 55)
    print(f"{'Overall':8} | {overall['ATM']:7.3f} | {overall['MID']:7.3f} | "
          f"{overall['WINGS']:7.3f} | {overall['OVERALL']:8.3f}\n")

    return None #per_maturity, overall

def plot_ivs_comparison(ivs_model, ivs_mkt, diff, log_money_grid, title, figure_size=(22, 5), elev=25, azim=-135, cmap="viridis", maturity_min=None, maturity_max=None):
    # Filter maturities based on range
    filtered_model = {}
    filtered_mkt = {}
    filtered_diff = {}
    
    for mat in ivs_model.keys():
        if maturity_min is not None and mat < maturity_min:
            continue
        if maturity_max is not None and mat > maturity_max:
            continue
        if mat in ivs_mkt and mat in diff:  # Only include if all three dicts have this maturity
            filtered_model[mat] = ivs_model[mat]
            filtered_mkt[mat] = ivs_mkt[mat]
            filtered_diff[mat] = diff[mat]
    
    maturities = sorted(filtered_model.keys())
    if not maturities:
        print("No maturities found in the specified range.")
        return None, None
        
    Z_model = np.vstack([filtered_model[t] for t in maturities]).T
    Z_mkt = np.vstack([filtered_mkt[t] for t in maturities]).T
    Z_diff = np.vstack([filtered_diff[t] for t in maturities]).T
    X, Y = np.meshgrid(maturities, log_money_grid)

    fig = plt.figure(figsize=figure_size)
    axes = []

    for i, (Z, title, zlab) in enumerate(
        [
            (Z_model, f"Model {title} Surface", f"{title} (model)"),
            (Z_mkt, f"Market {title} Surface", f"{title} (market)"),
            (Z_diff, "Model - Market Absolute(Diff)", f"{title} Difference in Base Point"),
        ],
        start=1,
    ):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True)
        ax.set_title(title)
        ax.set_xlabel("Time to Maturity")
        ax.set_ylabel("Log-moneyness")
        ax.set_zlabel(zlab)
        ax.view_init(elev=elev, azim=azim)
        ax.set_ylim(ax.get_ylim()[::-1])
        fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.06)
        axes.append(ax)

    fig.tight_layout()
    return fig, axes