import numpy as np
np.random.seed(42)
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.integrate import quad
import matplotlib.pyplot as plt
import time
import warnings
import multiprocessing as mp
from functools import partial

warnings.filterwarnings("ignore")

# ============================================================
# 高精度蒙特卡洛 (epsrel=1e-7) 30次模拟，带进度输出
# ============================================================

C_LIGHT = 299792.458
RD_FID = 147.09
INTEGRAL_EPSREL = 1e-7
INTEGRAL_LIMIT = 500

OM_NULL = 0.0177
H0_NULL = 76.25

def load_data():
    data = np.genfromtxt("pantheon+_data.txt", skip_header=1, usecols=(4,10,11))
    z, mu, err = data.T
    msk = (z > 0) & np.isfinite(mu)
    z, mu = z[msk], mu[msk]
    with open("Pantheon+SH0ES_STAT+SYS.cov", 'r') as f:
        n = int(f.readline())
    cov = np.loadtxt("Pantheon+SH0ES_STAT+SYS.cov", skiprows=1).reshape(n, n)
    cov = cov[np.ix_(msk, msk)]
    cov = (cov + cov.T) / 2
    cov = np.asarray(cov, dtype=float)
    cov += np.eye(cov.shape[0]) * 1e-8
    bao = {0.50: (18.65, 0.25), 0.70: (24.30, 0.30), 1.00: (31.80, 0.45)}
    return z, mu, cov, bao

def E_lcdm(z, Om):
    return np.sqrt(Om*(1+z)**3 + (1-Om))

def mu_lcdm(z, Om, H0):
    try:
        dc, _ = quad(lambda x: C_LIGHT / E_lcdm(x, Om), 0, z,
                     limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10
    dc /= H0
    return 5*np.log10(dc * (1+z)) + 25

def DM_H_z_lcdm(z, Om, H0):
    try:
        dc, _ = quad(lambda x: C_LIGHT / E_lcdm(x, Om), 0, z,
                     limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10, 1e10
    DM = dc / H0
    Hz = H0 * E_lcdm(z, Om)
    return DM, Hz

def rho_osc_at_z(z, A, a, b):
    try:
        val, _ = quad(lambda x: 3*A*np.exp(-a*x)*np.sin(b*x)/(1+x), 0, z,
                      limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        val = -np.inf
    if val < -100:
        val = -100
    return np.exp(val)

def mu_osc(z, A, a, b, Om, H0):
    def integrand(x):
        rho_x = rho_osc_at_z(x, A, a, b)
        if rho_x < 0:
            rho_x = 0.0
        return C_LIGHT / np.sqrt(Om*(1+x)**3 + (1-Om)*rho_x)
    try:
        dc, _ = quad(integrand, 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10
    return 5*np.log10((dc / H0) * (1+z)) + 25

def DM_H_z_osc(z, A, a, b, Om, H0):
    def integrand(x):
        rho_x = rho_osc_at_z(x, A, a, b)
        if rho_x < 0:
            rho_x = 0.0
        return C_LIGHT / np.sqrt(Om*(1+x)**3 + (1-Om)*rho_x)
    try:
        dc, _ = quad(integrand, 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10, 1e10
    DM = dc / H0
    rho_z = rho_osc_at_z(z, A, a, b)
    if rho_z < 0:
        rho_z = 0.0
    Hz = H0 * np.sqrt(Om*(1+z)**3 + (1-Om)*rho_z)
    return DM, Hz

def chi2_total(theta, z, mu, cov, bao, is_lcdm):
    if is_lcdm:
        Om, H0 = theta
        mu_model = np.array([mu_lcdm(zi, Om, H0) for zi in z])
    else:
        A, a, b, Om, H0 = theta
        if not (0 <= A <= 1.2 and 0 <= a <= 30 and 0 <= b <= 50):
            return 1e10
        mu_model = np.array([mu_osc(zi, A, a, b, Om, H0) for zi in z])
    if not np.isfinite(mu_model).all() or np.any(np.abs(mu_model) > 100):
        return 1e10
    resid = mu - mu_model
    try:
        cho = cho_factor(cov)
        chi2_sn = resid @ cho_solve(cho, resid)
    except:
        return 1e10
    chi2_bao = 0.0
    for z_bao, (obs_val, obs_err) in bao.items():
        if is_lcdm:
            D_M, H_z = DM_H_z_lcdm(z_bao, Om, H0)
        else:
            D_M, H_z = DM_H_z_osc(z_bao, A, a, b, Om, H0)
        if D_M > 1e9 or H_z < 1e-10 or not np.isfinite(D_M) or not np.isfinite(H_z):
            return 1e10
        term = (D_M**2) * (C_LIGHT * z_bao / H_z)
        if term <= 0:
            return 1e10
        D_V = term ** (1/3)
        model_val = D_V / RD_FID
        chi2_bao += ((model_val - obs_val) / obs_err) ** 2
    return chi2_sn + chi2_bao

def generate_mock_data(z, mu_lcdm_pred, cov):
    L = cholesky(cov, lower=True)
    noise = L @ np.random.randn(len(z))
    return mu_lcdm_pred + noise

def fit_one_simulation(seed, z, cov, bao, mu_lcdm_pred, Om_bounds):
    np.random.seed(seed)
    mu_sim = generate_mock_data(z, mu_lcdm_pred, cov)
    # ΛCDM 拟合
    def chi2_lcdm(theta):
        Om, H0 = theta
        mu_model = np.array([mu_lcdm(zi, Om, H0) for zi in z])
        resid = mu_sim - mu_model
        try:
            cho = cho_factor(cov)
            return resid @ cho_solve(cho, resid)
        except:
            return 1e10
    res_lcdm = minimize(chi2_lcdm, [0.3, 70.0],
                        bounds=[Om_bounds, (60,85)],
                        method='L-BFGS-B', options={'maxiter': 500})
    chi2_l = res_lcdm.fun
    # 振荡模型拟合
    def chi2_osc(theta):
        A, a, b, Om, H0 = theta
        if not (0 <= A <= 1.2 and 0 <= a <= 30 and 0 <= b <= 50):
            return 1e10
        mu_model = np.array([mu_osc(zi, A, a, b, Om, H0) for zi in z])
        resid = mu_sim - mu_model
        try:
            cho = cho_factor(cov)
            return resid @ cho_solve(cho, resid)
        except:
            return 1e10
    res_osc = minimize(chi2_osc, [0.0, 0.0, 20.0, 0.3, 70.0],
                       bounds=[(0,1.2), (0,30), (0,50), Om_bounds, (60,85)],
                       method='L-BFGS-B', options={'maxiter': 500})
    chi2_o = res_osc.fun
    return chi2_l - chi2_o

def run_montecarlo(n_sim=30, n_cores=None):
    z, mu, cov, bao = load_data()
    mu_lcdm_null = np.array([mu_lcdm(zi, OM_NULL, H0_NULL) for zi in z])
    Om_bounds = (0.01, 0.5)
    np.random.seed(42)
    seeds = np.random.randint(0, 2**30, size=n_sim)
    if n_cores is None:
        n_cores = mp.cpu_count()
    print(f"使用 {n_cores} 个进程并行模拟 {n_sim} 次 (高精度 epsrel=1e-7)")
    start_time = time.time()
    worker = partial(fit_one_simulation, z=z, cov=cov, bao=bao,
                     mu_lcdm_pred=mu_lcdm_null, Om_bounds=Om_bounds)
    with mp.Pool(processes=n_cores) as pool:
        results = []
        for i, delta in enumerate(pool.imap_unordered(worker, seeds), 1):
            results.append(delta)
            if i % 5 == 0 or i == n_sim:
                print(f"已完成 {i}/{n_sim} 次模拟, 当前 Δχ² = {delta:.2f}")
    elapsed = time.time() - start_time
    return np.array(results), elapsed, z, mu

if __name__ == "__main__":
    print("="*80)
    print("高精度蒙特卡洛模拟 (epsrel=1e-7, 30次)")
    print("零假设: Ωm=0.0177, H0=76.25")
    print("="*80)
    n_sim = 30
    delta_chi2_sim, elapsed, z, mu = run_montecarlo(n_sim=n_sim)
    real_delta = 147.28
    max_val = np.max(delta_chi2_sim)
    p_value = np.mean(delta_chi2_sim >= real_delta)
    print(f"\n模拟完成，耗时 {elapsed:.1f} 秒")
    print(f"最大 Δχ² = {max_val:.2f}")
    print(f"p值 = {p_value:.6f} (真实 Δχ²={real_delta})")
    # 保存结果
    np.save('delta_chi2_sim_30.npy', delta_chi2_sim)
    plt.hist(delta_chi2_sim, bins=20, alpha=0.7)
    plt.axvline(real_delta, color='r', linestyle='--')
    plt.xlabel('Δχ²')
    plt.ylabel('Frequency')
    plt.title('Monte Carlo (30 simulations, epsrel=1e-7)')
    plt.savefig('montecarlo_30.png')
    plt.show()