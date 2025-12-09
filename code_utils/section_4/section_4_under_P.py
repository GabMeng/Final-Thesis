import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def find_k(df, price_col : str = "price", alpha : float = 0.05):
    """
    Computes the jump size (in units of alpha) for each nonzero price movement in a time series DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing a price column.
        price_col (str, optional): Name of the column containing price data. Default is "price".
        alpha (float, optional): Tick size (minimum price increment). Default is 0.05.

    Returns:
        pd.DataFrame: DataFrame with additional columns:
            - "movement": price difference with previous row
            - "abs_jump_size": absolute movement divided by alpha
            - "abs_jump_size_round": rounded absolute jump size (as integer)
        The first row and rows with zero movement are excluded.
    """  
    df_f = df.copy()
    
    df_f["movement"] = df_f[price_col] - df_f[price_col].shift(1)
    df_f = df_f[df_f["movement"] != 0]
    df_f["abs_jump_size"] = np.abs(df_f["movement"]) / alpha
    df_f = df_f.iloc[1:]
    df_f["abs_jump_size_round"] = np.round(df_f["abs_jump_size"]).astype(int)

    
    return df_f

def N_alt_N_cont(df, k_col = "abs_jump_size_round", price_col = "price"):
    """
    Counts the number of alternations and continuations for each jump size k in a price series.

    Args:
        df (pd.DataFrame): Input DataFrame containing price and jump size columns.
        k_col (str, optional): Name of the column representing the jump size (default: "abs_jump_size_round").
        price_col (str, optional): Name of the column containing price data (default: "price").

    Returns:
        tuple:
            dict_alternation (dict): Dictionary where keys are jump sizes (as strings) and values are the count of alternations (sign change) for that jump size.
            dict_continuation (dict): Dictionary where keys are jump sizes (as strings) and values are the count of continuations (same sign) for that jump size.
    """   
    
    
    df_f = df.copy()
    
    dict_alternation = {}
    dict_continuation = {}
    for k in sorted(list(df[k_col].unique())):
        
        partial_df = df_f[df_f[k_col] == k]
        partial_df = partial_df[[price_col, k_col]]
        partial_df["movement"] = partial_df[price_col] - partial_df[price_col].shift(1)
        partial_df["sign"] = partial_df["movement"].apply(lambda x: 1 if x > 0 else -1)
        partial_df["a_vs_c"] = partial_df["sign"]*partial_df["sign"].shift(1) # if the sign is + this mean we have continuation otherwise alternation
        
        dict_alternation[str(k)] = sum(partial_df["a_vs_c"] == -1)
        dict_continuation[str(k)] = sum(partial_df["a_vs_c"] == 1)

        
    return dict_alternation, dict_continuation

def estimate_eta(d_a: dict, d_c: dict, m=None):
    """
    Estimates the aversion to price changes parameter eta_hat using alternation and continuation counts.
    Keys in d_a and d_c are strings "1", "2", ..., "m".
    If m is None, it is set to the maximum index present in both d_a and d_c.
    """

    # infer m if not provided
    if m is None:
        keys_a = set(int(k) for k in d_a.keys())
        keys_c = set(int(k) for k in d_c.keys())
        common_keys = keys_a & keys_c
        if not common_keys:
            raise ValueError("No overlapping observations between d_a and d_c.")
        m = max(common_keys)

    # Get all available keys (jump sizes) that exist in both dictionaries
    available_keys = set(d_a.keys()) & set(d_c.keys())
    available_k_values = sorted([int(k) for k in available_keys if int(k) <= m])
    
    if not available_k_values:
        return 0.0

    # Calculate total count for normalization using only available keys
    total_count = sum(d_a[str(j)] + d_c[str(j)] for j in available_k_values)
    if total_count == 0:
        return 0.0

    total = 0.0
    for k in available_k_values:
        key = str(k)

        Na = d_a[key]
        Nc = d_c[key]

        lam_num = Na + Nc
        if lam_num == 0:
            continue
            
        lam = lam_num / total_count

        if Na == 0:
            if Nc > 0:
                return 1.0
            else:
                continue

        u = 0.5 * (k * (Nc / Na - 1) + 1)
        total += lam * u

    return max(0.0, min(1.0, total))

def estimate_integrated_vol(df, eta_hat, price_col = "price", tick_size = 0.05):
    """
    Estimates the integrated variance using the price series.

    Args:
        df (pd.DataFrame): Input DataFrame containing the price series.
        eta_hat (float): Estimated price change aversion noise parameter.
        price_col (str, optional): Name of the column containing price data. Default is "price".
        tick_size (float, optional): Tick size (minimum price increment). Default is 0.05.

    Returns:
        float: Estimated integrated volatility (realized variance) of the adjusted price series.
    """
    
    df_f = df.copy()
    df_f["hat_X_t"] = df_f[price_col] - tick_size*(0.5 - eta_hat)*np.sign((df_f[price_col] - df_f[price_col].shift(1)))
    ln_ret = (np.log(df_f["hat_X_t"]) - np.log(df_f["hat_X_t"].shift(1)))
    ln_ret = ln_ret.dropna()
    integrated_vol = np.sum(ln_ret**2)
    
    return integrated_vol

def find_daily_RV(df, eta_daily = False):
    
    # we first estimate eta for the whole price process.
    df_f = df.copy() #feed in df with timestamp index and "price" column
    df_movement_k_alpha = find_k(df_f)
    d_alt_whole, d_cont_whole = N_alt_N_cont(df_movement_k_alpha)
    eta_whole = estimate_eta(d_alt_whole, d_cont_whole)
    
    # we find the daily Integrated variance. 
    out = []
    for d in df_f.index.normalize().unique():
        daily_df = df_f[df_f.index.normalize() == d]
        if eta_daily:
            try:
                df_loop = daily_df.copy()
                df_movement_k_alpha_loop = find_k(df_loop)
                d_alt_whole_loop, d_cont_whole_loop = N_alt_N_cont(df_movement_k_alpha_loop)
                eta_partition = estimate_eta(d_alt_whole_loop, d_cont_whole_loop)
                sigma_daily = estimate_integrated_vol(df_loop, eta_partition)
            except:
                continue
        else:
            sigma_daily = estimate_integrated_vol(daily_df, eta_whole)

        row = {"date":d, "realized_variance" : sigma_daily}
        out.append(row)
        
    df_out = pd.DataFrame(out).sort_values("date").reset_index(drop=True)
    df_out = df_out[df_out["realized_variance"] > 0] # filter out 0 vol day, ie weekend and holydays
    return df_out


def m_q_del_raw(q, lag, sig):
    log_val = np.log(sig)
    log_diff = log_val[lag:] - log_val[:-lag]
    abs_log_diff = np.abs(log_diff)
    q_power = np.pow(abs_log_diff, q)
    mean = np.mean(q_power)
    return mean 

def m_q_more_lag(q, lag_list, sig):
    return np.array([m_q_del_raw(q, lag, sig) for lag in lag_list])


def m_q_over_lags(q, lags, sig, average_offsets=False):
    """
    Compute m_q(Δ) for each lag in lags.

    Parameters
    ----------
    q : float
        Moment order.
    lags : array-like
        List/array of lag values to compute over.
    log_sig : array-like
        Log volatility time series (1D array).
    average_offsets : bool, default=False
        If False, use standard overlapping differences:
            log_sig[lag:] - log_sig[:-lag]
        If True, compute m_q for each offset 0...(lag-1) and return the mean.

    Returns
    -------
    np.ndarray
        Array of m_q values, one per lag.
    """
    results = []
    log_sig = np.log(sig)

    for lag in lags:
        if lag <= 0 or lag >= len(log_sig):
            results.append(np.nan)
            continue

        if not average_offsets:
            diffs = log_sig[lag:] - log_sig[:-lag]
            val = np.mean(np.abs(diffs) ** q)
            results.append(val)
        else:
            vals = []
            for offset in range(lag):
                sub = log_sig[offset:]
                if len(sub) <= lag:
                    continue
                diffs = sub[lag:] - sub[:-lag]
                vals.append(np.mean(np.abs(diffs) ** q))
            results.append(np.mean(vals) if vals else np.nan)

    return np.array(results)

def reg_and_print(q_vec, lags_list, sig, plot=True, average = True):
    zeta_q = np.empty(len(q_vec)) 
    
    if plot:
        plt.figure(figsize=(8, 5))

    for i, q in enumerate(q_vec):
        m_vals = m_q_over_lags(q, lags_list, sig, average_offsets=average)
        mask = np.isfinite(m_vals) & (m_vals > 0)
        lx = np.log(lags_list[mask])
        ly = np.log(m_vals[mask])
        
        slope, intercept = np.polyfit(lx, ly, 1)
        zeta_q[i] = slope

        if plot:
            plt.scatter(lx, ly, s=10, label=None)
            plt.plot(lx, slope * lx + intercept, linewidth=3, label=f"p = {q}")

    if plot:
        plt.xlabel("ln Δ")
        plt.ylabel("ln m(p, Δ)")
        plt.legend(title="Orders p")
        plt.tight_layout()
        plt.grid(True, alpha = 0.5)
        plt.show()
        
    return zeta_q

def estimate_H(q_vec, zeta_q, force_origin=False, plot=True):
    """
    Estimate H from the relation zeta(q) ≈ H * q.

    Parameters
    ----------
    q_vec : array-like
        List/array of q values (e.g. [0.5, 1, 1.5, 2, 3])
    zeta_q : array-like
        Corresponding zeta(q) values (slopes from log-log regressions)
    force_origin : bool, default=False
        If True, fit regression without intercept (ζ(q) = H q).
    plot : bool, default=True
        If True, plot ζ(q) vs q with fitted line.

    Returns
    -------
    H : float
        Estimated slope (H)
    intercept : float
        Intercept from regression (0 if force_origin=True)
    """
    q_vec = np.array(q_vec).reshape(-1, 1)
    zeta_q = np.array(zeta_q)

    if force_origin:
        reg = LinearRegression(fit_intercept=False).fit(q_vec, zeta_q)
    else:
        reg = LinearRegression().fit(q_vec, zeta_q)

    H = reg.coef_[0]
    intercept = reg.intercept_ if not force_origin else 0.0
    pred = reg.predict(q_vec)

    if plot:
        plt.figure(figsize=(6, 4))
        plt.scatter(q_vec, zeta_q, color="blue", s=3, label="Estimated Ψ(p)")
        plt.plot(q_vec, pred, color="red", lw=2,
                 label=f"Fit: Ψ(p) = ({H:.3f})·p")
        plt.xlabel("p")
        plt.ylabel("Ψ(p)")
        plt.legend()
        plt.tight_layout()
        plt.grid(True, alpha = 0.5)
        plt.show()

    return float(H), float(intercept)