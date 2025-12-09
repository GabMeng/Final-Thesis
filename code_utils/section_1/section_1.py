import numpy as np
import matplotlib.pyplot as plt

def davies_harte(T, N, H):
    '''
    Generates sample paths of fractional Brownian Motion using the Davies Harte method
    
    Credit : Justin Yu, M.S. Financial Engineering, Stevens Institute of Technology
    
    args:
        T:      length of time (in years)
        N:      number of time steps within timeframe
        H:      Hurst parameter
    '''
    gamma = lambda k,H: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))  
    g = [gamma(k,H) for k in range(0,N)];    r = g + [0] + g[::-1][0:N-1]

    # Step 1 (eigenvalues)
    j = np.arange(0,2*N);   k = 2*N-1
    lk = np.fft.fft(r*np.exp(2*np.pi*1j*k*j*(1/(2*N))))[::-1]

    # Step 2 (get random variables)
    Vj = np.zeros((2*N,2), dtype=np.complex128); 
    Vj[0,0] = np.random.standard_normal();  Vj[N,0] = np.random.standard_normal()
    
    for i in range(1,N):
        Vj1 = np.random.standard_normal();    Vj2 = np.random.standard_normal()
        Vj[i][0] = Vj1; Vj[i][1] = Vj2; Vj[2*N-i][0] = Vj1;    Vj[2*N-i][1] = Vj2
    
    # Step 3 (compute Z)
    wk = np.zeros(2*N, dtype=np.complex128)   
    wk[0] = np.sqrt((lk[0]/(2*N)))*Vj[0][0];          
    wk[1:N] = np.sqrt(lk[1:N]/(4*N))*((Vj[1:N].T[0]) + (1j*Vj[1:N].T[1]))       
    wk[N] = np.sqrt((lk[0]/(2*N)))*Vj[N][0]       
    wk[N+1:2*N] = np.sqrt(lk[N+1:2*N]/(4*N))*(np.flip(Vj[1:N].T[0]) - (1j*np.flip(Vj[1:N].T[1])))
    
    Z = np.fft.fft(wk);     fGn = Z[0:N] 
    fBm = np.cumsum(fGn)*(N**(-H))
    fBm = (T**H)*(fBm)
    path = np.array([0] + list(fBm))
    return path

def plot_fbm_path(fbm_path, H, T=1, N=252, title=None):
    """
    Plot a fractional Brownian motion path
    
    Args:
        fbm_path: Array of fBM values from davies_harte function
        H: Hurst parameter used to generate the path
        T: Time horizon (default 1 year)
        N: Number of time steps (default 252)
        title: Custom title for the plot
    """
    # Create time grid
    time_grid = np.linspace(0, T, len(fbm_path))
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_grid, fbm_path, linewidth=1.5, color="blue")
    plt.xlabel("Time (years)")
    plt.ylabel("fBM Value")
    
    if title is None:
        title = f"Fractional Brownian Motion Path (H = {H})"
    plt.title(title)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    
def gamma_k(k, H):
    return 0.5*(np.abs(k+1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k-1)**(2*H))

ks = np.arange(-50, 50)

H_anti_pers = np.array([0.05, 0.15, 0.25, 0.35, 0.45])
H_pers = H_anti_pers+0.5
H_brownian = 0.5  # Standard Brownian motion

# Evaluate gamma_k for all k values and H parameters
gamma_results_anti_pers = []
gamma_results_pers = []
gamma_results_brownian = [gamma_k(k, H_brownian) for k in ks]

# Calculate gamma_k for anti-persistent H values (H < 0.5)
for H in H_anti_pers:
    gamma_values = [gamma_k(k, H) for k in ks]
    gamma_results_anti_pers.append(gamma_values)

# Calculate gamma_k for persistent H values (H > 0.5)
for H in H_pers:
    gamma_values = [gamma_k(k, H) for k in ks]
    gamma_results_pers.append(gamma_values)

# Create three plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Anti-persistent case (H < 0.5)
for i, H in enumerate(H_anti_pers):
    ax1.plot(ks, gamma_results_anti_pers[i], label=f"H = {H}", linewidth=2)

ax1.set_xlabel("k")
ax1.set_ylabel("γ(k)")
ax1.set_title("Gamma Function for Anti-Persistent Cases (H < 0.5)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Standard Brownian motion (H = 0.5)
ax2.plot(ks, gamma_results_brownian, label=f"H = {H_brownian}", linewidth=2, color="blue")
ax2.set_xlabel("k")
ax2.set_ylabel("γ(k)")
ax2.set_title("Gamma Function for Standard Brownian Motion (H = 0.5)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Persistent case (H > 0.5)
for i, H in enumerate(H_pers):
    ax3.plot(ks, gamma_results_pers[i], label=f"H = {H}", linewidth=2)

ax3.set_xlabel("k")
ax3.set_ylabel("γ(k)")
ax3.set_title("Gamma Function for Persistent Cases (H > 0.5)")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()