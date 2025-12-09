import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

def interp_risk_free(option_chain, risk_free_curve):
    """
    Interpolates risk-free discount rates for option maturities using linear interpolation 
    on the logarithm of discount factors from a risk-free rate curve.
    
    Args:
        option_chain (pd.DataFrame): DataFrame containing option data with a "time_to_maturity" 
            column representing the time to expiration in years.
        risk_free_curve (pd.DataFrame): DataFrame containing the risk-free rate curve with 
            columns:
            - "maturity": Time to maturity in years (sorted ascending)
            - "df": Discount factors corresponding to each maturity
    
    Returns:
        pd.DataFrame: A copy of the input option_chain DataFrame with an additional 
            "discount_rate" column containing the interpolated discount factors for each 
            option's time to maturity.

    """
    
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
    
    
    
#test_df = pd.DataFrame()
#test_df["time_to_maturity"] = [0.134, 0.77, 0.75, 2.2]

#test_df = interp_risk_free(test_df, risk_free_rate_estr_ois_15_09)
#test_df
    
# controllato con desmos. 

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


def clean_options(df,
                  min_iv=1e-3, max_iv=10.0,
                  min_volume=0, min_oi=0,
                  min_log_k = -1, max_log_k = 5):
    """
    Filters and cleans option chain data by removing invalid quotes and applying 
    liquidity and moneyness constraints.
    
    Args:
        df (pd.DataFrame): DataFrame containing option data with columns:
            - "time_to_maturity": Time to expiration in years
            - "ivol": Implied volatility values
            - "moneyness": Log-moneyness values (ln(K/F))
            - "volume": Trading volume
            - "oi": Open interest
        min_iv (float, optional): Minimum implied volatility threshold. Defaults to 1e-3.
        max_iv (float, optional): Maximum implied volatility threshold. Defaults to 10.0.
        min_volume (int, optional): Minimum volume threshold. Defaults to 0.
        min_oi (int, optional): Minimum open interest threshold. Defaults to 0.
        min_log_k (float, optional): Minimum log-moneyness threshold. Defaults to -1.
        max_log_k (float, optional): Maximum log-moneyness threshold. Defaults to 5.
    
    Returns:
        pd.DataFrame: Filtered copy of the input DataFrame containing only options
            that meet all specified criteria.
    
    """

    # --- Step 1: drop non-quotes / invalid IVs ---
    df = df.copy()
    df = df[(df["time_to_maturity"] > 0) & (df["time_to_maturity"] < 1.5)]                                      
    df = df[(df["ivol"] >= min_iv) & (df["ivol"] <= max_iv)]    # sanity check on IVs
    df = df[(df["moneyness"] >= min_log_k) & (df["moneyness"] <= max_log_k)] 
    # --- Step 2: absolute volume / OI filters ---
    df = df[(df["volume"] >= min_volume) & (df["oi"] >= min_oi)]

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

    # Optional soft penalty to help guide feasibility (kept zero by default)
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
    Calibrates SVI model parameters for multiple maturity slices with calendar arbitrage 
    constraints enforced during optimization. Uses sequential constrained optimization 
    (SLSQP) to ensure total variance is non-decreasing in time.
    
    The function sequentially fits SVI parameters for each maturity T_i, enforcing the 
    calendar arbitrage constraint: w(k, T_i) >= w(k, T_{i-1}) + eps_calendar for all 
    strikes k in the constraint grid.
    
    Args:
        df (pd.DataFrame): DataFrame containing option data with columns specified by 
            mon_col, time_col, and ivol_col.
        min_ttm (float): Minimum time to maturity to include in calibration.
        max_ttm (float): Maximum time to maturity to include in calibration.
        number_of_mat (int, optional): Maximum number of maturities to use. If specified, 
            selects the most liquid maturities. Defaults to None (use all available).
        mon_col (str, optional): Column name for moneyness values (K/F). Defaults to "moneyness".
        time_col (str, optional): Column name for time to maturity in years. Defaults to "time_to_maturity".
        ivol_col (str, optional): Column name for implied volatility values. Defaults to "ivol".
        plotting_smile (bool, optional): If True, plots individual constrained smile fits. 
            Defaults to False.
        k_dimension (int, optional): Number of points for individual smile plotting grids. 
            Defaults to 100.
        plotting_surface (bool, optional): If True, displays 3D surface plots of IV and 
            total variance. Defaults to True.
        ks_number (int, optional): Number of strike points for surface evaluation grid. 
            Defaults to 100.
        manual_ks_grid (array-like, optional): Custom strike grid for surface evaluation. 
            Overrides ks_number if provided. Defaults to None.
        eps_calendar (float, optional): Minimum increment for calendar constraint enforcement. 
            Set to small positive value for strict inequality. Defaults to 0.0.
        constraint_density (int, optional): Number of points in the constraint grid for 
            calendar arbitrage enforcement. Defaults to 80.
        soft_penalty_lambda (float, optional): Weight for soft penalty term to help with 
            constraint feasibility. Defaults to 0.0.
    
    Returns:
        tuple: A tuple containing:
            - results (dict): Dictionary mapping each maturity to its fitted SVI parameters [a, b, ρ, m, σ].
            - strikes (dict): Dictionary mapping each maturity to its [min_strike, max_strike] range.
            - ks_grid (np.ndarray): Strike grid used for surface evaluation.
            - ivol_surface (np.ndarray): 2D array of implied volatilities with shape (n_maturities, n_strikes).
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

    return results, strikes, ks_grid, ivol_surface

def check_RND(params, k_grid, plot = True, tollerance = 1e-6):
    """
    Validates the risk-neutral density (RND) implied by SVI parameters by checking 
    if the density function is non-negative across a given strike grid.
    
    The function computes the risk-neutral density using the SVI total variance 
    function and checks the no-arbitrage condition by verifying that g(k) >= 0,
    where g(k) is derived from the Dupire formula for local volatility.
    
    Args:
        params (array-like): SVI parameters [a, b, ρ, m, σ] where:
        k_grid (array-like): Grid of log-moneyness values to evaluate the density.
        plot (bool, optional): If True, displays a plot of the risk-neutral density. 
            Defaults to True.
        tollerance (float, optional): Numerical tolerance for checking non-negativity 
            of g(k). Defaults to 1e-6.
    
    Returns:
        tuple: A tuple containing:
            - g_val (np.ndarray): Values of the g(k) function across the strike grid.
            - p_val (np.ndarray): Risk-neutral density values p(k) across the strike grid.
            - all_g_positive (bool): True if g(k) + tolerance >= 0 for all k in the grid,
              indicating a valid (arbitrage-free) risk-neutral density.
    """
    a, b, rho, m, sigma = params
    def w(k):
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    def w_prime(k):
        return b * rho - b * (k - m) / np.sqrt((k - m)**2 + sigma**2)
    def w_second(k):
        s_square = sigma**2
        return (b*s_square) / ((k-m)**2 + s_square)**(3/2)
    def g(k): 
        wk = w(k)
        w_p = w_prime(k)
        w_s = w_second(k)
        y = (1-((k*w_p)/(2*wk)))**2 - ((w_p**2)/4)*((1/wk) + 0.25) + 0.5*(w_s)
        return y 
    def d_plus(k):
        return (-k)/np.sqrt(w(k)) + 0.5*np.sqrt(w(k))
    def d_minus(k):
        return (-k)/np.sqrt(w(k)) - 0.5*np.sqrt(w(k))
    
    def p(k):
        p_k = ((g(k))/(np.sqrt(2*np.pi*w(k))))*np.exp(-0.5*d_minus(k))
        return p_k
    
    g_val = np.array([g(k) for k in k_grid])
    p_val = np.array([p(k) for k in k_grid])
    
    tol = tollerance
    all_g_positive = np.all(g_val + tol >= 0)

    if plot:
        plt.figure(figsize=(8,5))
        plt.plot(k_grid, p_val, label=r"$p_k$")
        plt.axhline(0, linestyle="--", linewidth=0.8)
        plt.xlabel("k")
        plt.ylabel(r"$p_k$")
        plt.title(f"p_k over k, d_plus asympote: {d_plus(1e6)}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return g_val, p_val, all_g_positive  