import numpy as np
import multiprocessing as mp
from functools import partial
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.integrate import quad
import matplotlib.pyplot as plt
import time
import warnings
import os

warnings.filterwarnings("ignore")

# ============================================================
# 常数与模型函数（优化精度版）
# ============================================================

C_LIGHT = 299792.458
RD_FID = 147.09   # DESI fiducial sound horizon

# 全局积分精度设置（可调，速度与精度的平衡）
INTEGRAL_EPSREL = 1e-5   # 原为1e-6，提高速度
INTEGRAL_LIMIT = 150     # 原为200

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

    bao = {
        0.50: (18.65, 0.25),
        0.70: (24.30, 0.30),
        1.00: (31.80, 0.45)
    }
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
        dc, _ = quad(integrand, 0, z,
                     limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
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
        dc, _ = quad(integrand, 0, z,
                     limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
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

def fit_one_simulation(mu_sim, z, cov, bao, Om_bounds):
    """单个模拟数据集的拟合，返回Δχ²（优化迭代次数）"""
    opt_options = {'maxiter': 800, 'finite_diff_rel_step': 1e-5}
    # ΛCDM
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
                        method='L-BFGS-B', options=opt_options)
    chi2_l = res_lcdm.fun

    # 振荡模型
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
                       method='L-BFGS-B', options=opt_options)
    chi2_o = res_osc.fun
    return chi2_l - chi2_o

def generate_mock_data(z, mu_lcdm_pred, cov):
    L = cholesky(cov, lower=True)
    noise = L @ np.random.randn(len(z))
    return mu_lcdm_pred + noise

def run_single_simulation(seed, z, cov, bao, Om_bounds, mu_lcdm_pred, output_file):
    np.random.seed(seed)
    mu_sim = generate_mock_data(z, mu_lcdm_pred, cov)
    delta = fit_one_simulation(mu_sim, z, cov, bao, Om_bounds)
    # 实时写入文件（每个结果一行，避免丢失）
    with open(output_file, 'a') as f:
        f.write(f"{seed}, {delta}\n")
    return delta

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    # 加载真实数据
    z, mu_real, cov, bao = load_data()

    # 零假设参数（固定rd的ΛCDM最佳拟合）
    Om_null = 0.0177
    H0_null = 76.25

    # 预计算ΛCDM预测（用于生成模拟数据）
    mu_lcdm_null = np.array([mu_lcdm(zi, Om_null, H0_null) for zi in z])

    Om_bounds = (0.01, 0.5)
    n_sim = 100          # 可修改为50或100
    output_file = "montecarlo_results.csv"

    # 清空旧结果文件
    if os.path.exists(output_file):
        os.remove(output_file)

    # 准备种子列表
    np.random.seed(42)
    seeds = np.random.randint(0, 2**30, size=n_sim)

    # 并行计算
    n_cores = mp.cpu_count()
    print(f"检测到CPU核心数: {n_cores}，将使用 {n_cores} 个进程并行计算")
    print(f"开始蒙特卡洛模拟，共 {n_sim} 次，零假设参数: Ωm={Om_null}, H0={H0_null}")
    print(f"积分精度: epsrel={INTEGRAL_EPSREL}, limit={INTEGRAL_LIMIT}")
    print(f"优化器迭代上限: 800")
    start_time = time.time()

    # 使用 partial 固定参数，并传递输出文件路径
    partial_func = partial(run_single_simulation,
                           z=z, cov=cov, bao=bao,
                           Om_bounds=Om_bounds,
                           mu_lcdm_pred=mu_lcdm_null,
                           output_file=output_file)

    with mp.Pool(processes=n_cores) as pool:
        # 使用 imap_unordered 以便实时显示进度
        results = []
        for i, delta in enumerate(pool.imap_unordered(partial_func, seeds), 1):
            results.append(delta)
            if i % 10 == 0:
                print(f"已完成 {i}/{n_sim} 次模拟, 当前 Δχ² = {delta:.2f}")
    elapsed = time.time() - start_time
    print(f"模拟完成，耗时 {elapsed:.1f} 秒")

    delta_chi2_sim = np.array(results)

    # 保存完整数组
    np.save('delta_chi2_sim.npy', delta_chi2_sim)

    # 统计量
    real_delta = 147.28
    max_val = np.max(delta_chi2_sim)
    min_val = np.min(delta_chi2_sim)
    mean_val = np.mean(delta_chi2_sim)
    std_val = np.std(delta_chi2_sim)
    median_val = np.median(delta_chi2_sim)
    p99 = np.percentile(delta_chi2_sim, 99)
    p_value = np.mean(delta_chi2_sim >= real_delta)

    print("\n========== 模拟结果统计 ==========")
    print(f"最大 Δχ² = {max_val:.2f}")
    print(f"最小 Δχ² = {min_val:.2f}")
    print(f"平均 Δχ² = {mean_val:.2f}")
    print(f"标准差 Δχ² = {std_val:.2f}")
    print(f"中位数 Δχ² = {median_val:.2f}")
    print(f"99% 分位数 = {p99:.2f}")
    print(f"真实 Δχ² = {real_delta:.2f}")
    print(f"p值 (Δχ² >= {real_delta}) = {p_value:.6f}")
    print("==================================\n")

    # 绘图
    plt.figure(figsize=(8,5))
    bins = np.linspace(min_val, max(max_val, real_delta), 50)
    plt.hist(delta_chi2_sim, bins=bins, density=True, alpha=0.7, color='gray', edgecolor='black')
    plt.axvline(real_delta, color='red', linestyle='--', linewidth=2, label=f'Real Δχ² = {real_delta}')
    plt.xlabel('Δχ² (oscillatory minus ΛCDM)')
    plt.ylabel('Probability density')
    plt.title(f'Monte Carlo Simulation (n={n_sim}): Null hypothesis of smooth ΛCDM')
    plt.legend()
    plt.savefig('montecarlo_deltachi2.png', dpi=200)
    plt.show()

    print(f"详细结果已保存到 {output_file} 和 delta_chi2_sim.npy")