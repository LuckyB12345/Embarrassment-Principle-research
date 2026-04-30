import numpy as np
np.random.seed(42)
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.integrate import quad
import time
import warnings
import multiprocessing as mp
from functools import partial

warnings.filterwarnings("ignore")

# ============================================================
# 修正版蒙特卡洛 (低精度, 10次, α∈[-12,30])
# 零假设: ΛCDM (Ωm=0.0177, H0=76.25)
# BAO 模拟值基于理论 ΛCDM 计算，而非观测值
# ============================================================

C_LIGHT = 299792.458
RD_FID = 147.09
# 低精度设置
INTEGRAL_EPSREL = 1e-5
INTEGRAL_LIMIT = 200

# 零假设参数
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

    # 真实观测的 BAO 误差 (用于模拟噪声)
    bao_err = {0.50: 0.25, 0.70: 0.30, 1.00: 0.45}
    return z, mu, cov, bao_err

# ---------- ΛCDM 理论 BAO 值计算 ----------
def E_lcdm(z, Om):
    return np.sqrt(Om*(1+z)**3 + (1-Om))

def DM_H_z_lcdm(z, Om, H0):
    try:
        dc, _ = quad(lambda x: C_LIGHT / E_lcdm(x, Om), 0, z,
                     limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10, 1e10
    DM = dc / H0
    Hz = H0 * E_lcdm(z, Om)
    return DM, Hz

def compute_bao_theory(z_bao, Om, H0):
    """计算给定 ΛCDM 参数下的理论 D_V/r_d 值"""
    D_M, H_z = DM_H_z_lcdm(z_bao, Om, H0)
    term = (D_M**2) * (C_LIGHT * z_bao / H_z)
    if term <= 0:
        return 1e10
    D_V = term ** (1/3)
    return D_V / RD_FID

# ---------- 振荡模型所用函数 (与主脚本一致, α边界 -12~30) ----------
def mu_lcdm(z, Om, H0):
    try:
        dc, _ = quad(lambda x: C_LIGHT / E_lcdm(x, Om), 0, z,
                     limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10
    dc /= H0
    return 5*np.log10(dc * (1+z)) + 25

def rho_osc_at_z(z, A, a, b):
    try:
        val, _ = quad(lambda x: 3*A*np.exp(-a*x)*np.sin(b*x)/(1+x), 0, z,
                      limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        val = -np.inf
    if val < -300:
        val = -300
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

def chi2_total(theta, z, mu, cov, bao_obs, is_lcdm):
    if is_lcdm:
        Om, H0 = theta
        mu_model = np.array([mu_lcdm(zi, Om, H0) for zi in z])
    else:
        A, a, b, Om, H0 = theta
        if not (0 <= A <= 1.2 and -12 <= a <= 30 and 0 <= b <= 50):
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
    for z_bao, (obs_val, obs_err) in bao_obs.items():
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

def generate_mock_data(z, mu_lcdm_pred, cov, bao_theory, bao_err):
    """生成模拟数据：SN 基于协方差，BAO 基于理论值+高斯噪声"""
    L = cholesky(cov, lower=True)
    noise_sn = L @ np.random.randn(len(z))
    mu_sim = mu_lcdm_pred + noise_sn

    bao_sim = {}
    for z_bao, theory_val in bao_theory.items():
        err = bao_err[z_bao]
        bao_sim[z_bao] = (theory_val + np.random.randn() * err, err)
    return mu_sim, bao_sim

def fit_one_simulation(seed, z, cov, bao_err, mu_lcdm_pred, bao_theory, Om_bounds):
    np.random.seed(seed)
    mu_sim, bao_sim = generate_mock_data(z, mu_lcdm_pred, cov, bao_theory, bao_err)

    # ΛCDM 拟合
    def chi2_lcdm(theta):
        Om, H0 = theta
        return chi2_total([Om, H0], z, mu_sim, cov, bao_sim, is_lcdm=True)
    res_lcdm = minimize(chi2_lcdm, [0.3, 70.0],
                        bounds=[Om_bounds, (60,85)],
                        method='L-BFGS-B', options={'maxiter': 300, 'disp': False})
    chi2_l = res_lcdm.fun

    # 振荡模型拟合
    def chi2_osc(theta):
        A, a, b, Om, H0 = theta
        return chi2_total([A, a, b, Om, H0], z, mu_sim, cov, bao_sim, is_lcdm=False)
    res_osc = minimize(chi2_osc, [0.0, -5.0, 18.0, 0.3, 70.0],
                       bounds=[(0,1.2), (-12,30), (0,50), Om_bounds, (60,85)],
                       method='L-BFGS-B', options={'maxiter': 300, 'disp': False})
    chi2_o = res_osc.fun
    return chi2_l - chi2_o

def run_montecarlo(n_sim=10):
    # 加载真实红移、协方差、BAO 误差
    z, mu, cov, bao_err = load_data()
    # 计算零假设下的 ΛCDM 理论距离模量
    mu_lcdm_null = np.array([mu_lcdm(zi, OM_NULL, H0_NULL) for zi in z])
    # 计算零假设下的 BAO 理论值 (D_V/r_d)
    bao_z = list(bao_err.keys())
    bao_theory = {z_bao: compute_bao_theory(z_bao, OM_NULL, H0_NULL) for z_bao in bao_z}
    Om_bounds = (0.01, 0.5)

    np.random.seed(42)
    seeds = np.random.randint(0, 2**30, size=n_sim)
    n_cores = mp.cpu_count()
    print(f"使用 {n_cores} 个进程并行模拟 {n_sim} 次 (低精度, α∈[-12,30])")
    print("零假设: ΛCDM (Ωm=0.0177, H0=76.25)")
    print("BAO 理论值已基于零假设计算，观测噪声模拟正确。")
    start = time.time()

    worker = partial(fit_one_simulation,
                     z=z, cov=cov, bao_err=bao_err,
                     mu_lcdm_pred=mu_lcdm_null,
                     bao_theory=bao_theory,
                     Om_bounds=Om_bounds)

    with mp.Pool(processes=n_cores) as pool:
        results = []
        for i, delta in enumerate(pool.imap_unordered(worker, seeds), 1):
            results.append(delta)
            print(f"已完成 {i}/{n_sim} 次, 最近 Δχ² = {delta:.2f}")
            # 每2次自动保存一次中间结果（可选）
            if i % 2 == 0:
                np.save('temp_delta_low.npy', np.array(results))

    elapsed = time.time() - start
    delta_chi2_sim = np.array(results)
    return delta_chi2_sim, elapsed

if __name__ == "__main__":
    print("="*80)
    print("修正版蒙特卡洛 (低精度, 10次, α∈[-12,30])")
    print("真实观测 Δχ² = 459.34")
    print("="*80)
    n_sim = 10
    delta_chi2_sim, elapsed = run_montecarlo(n_sim=n_sim)

    real_delta = 459.34
    max_val = np.max(delta_chi2_sim)
    p_value = np.mean(delta_chi2_sim >= real_delta)

    print(f"\n✅ 完成，耗时 {elapsed:.1f} 秒")
    print(f"模拟 Δχ² 最大值 = {max_val:.2f}")
    print(f"p-value = {p_value:.6f} (基于 {n_sim} 次模拟)")
    print(f"真实 Δχ² = {real_delta}")

    np.save('delta_chi2_low_10.npy', delta_chi2_sim)