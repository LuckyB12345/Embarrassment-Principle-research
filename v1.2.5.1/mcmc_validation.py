import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import quad
import emcee
import corner
import multiprocessing as mp
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ========== 常量 (临时降低积分精度加速) ==========
C_LIGHT = 299792.458
RD_FID = 147.09
INTEGRAL_EPSREL = 1e-4      # 比正式 1e-5 更低，大幅加速 (误差可接受)
INTEGRAL_LIMIT = 500

# ========== 数据加载 ==========
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
    cov += np.eye(cov.shape[0]) * 1e-8
    bao = {0.50: (18.65, 0.25), 0.70: (24.30, 0.30), 1.00: (31.80, 0.45)}
    return z, mu, cov, bao

# ========== ΛCDM 函数（保留，但实际不需要）==========
def E_lcd(z, Om):
    return np.sqrt(Om*(1+z)**3 + (1-Om))

def mu_lcd(z, Om, H0):
    try:
        dc, _ = quad(lambda x: C_LIGHT / E_lcd(x, Om), 0, z,
                     limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10
    return 5*np.log10((dc / H0) * (1+z)) + 25

def DM_H_z_lcd(z, Om, H0):
    try:
        dc, _ = quad(lambda x: C_LIGHT / E_lcd(x, Om), 0, z,
                     limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10, 1e10
    DM = dc / H0
    Hz = H0 * E_lcd(z, Om)
    return DM, Hz

# ========== 振荡模型 ==========
def one_plus_w(z, A, b):
    return A * np.sin(b * z) / (1+z)**2

def rho_de_integrand(x, A, b):
    return 3 * one_plus_w(x, A, b) / (1+x)

def rho_osc_at_z(z, A, b):
    try:
        val, _ = quad(lambda x: rho_de_integrand(x, A, b), 0, z,
                      limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        val = -300.0
    val = np.clip(val, -300, 300)
    return np.exp(val)

def mu_osc(z, A, b, Om, H0):
    def integrand_dc(x):
        rho_x = rho_osc_at_z(x, A, b)
        rho_x = np.clip(rho_x, 0.0, 1e6)
        return C_LIGHT / np.sqrt(Om*(1+x)**3 + (1-Om)*rho_x)
    try:
        dc, _ = quad(integrand_dc, 0, z,
                     limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10
    return 5*np.log10((dc / H0) * (1+z)) + 25

def DM_H_z_osc(z, A, b, Om, H0):
    def integrand_dc(x):
        rho_x = rho_osc_at_z(x, A, b)
        rho_x = np.clip(rho_x, 0.0, 1e6)
        return C_LIGHT / np.sqrt(Om*(1+x)**3 + (1-Om)*rho_x)
    try:
        dc, _ = quad(integrand_dc, 0, z,
                     limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10, 1e10
    DM = dc / H0
    rho_z = rho_osc_at_z(z, A, b)
    rho_z = np.clip(rho_z, 0.0, 1e6)
    Hz = H0 * np.sqrt(Om*(1+z)**3 + (1-Om)*rho_z)
    return DM, Hz

# ========== Planck 压缩似然 ==========
def planck2018_chi2(Om, H0):
    Om_p, H0_p = 0.3111, 67.66
    sig_Om, sig_H0 = 0.0056, 0.42
    corr = 0.18
    chi2 = ((Om - Om_p)/sig_Om)**2 + ((H0 - H0_p)/sig_H0)**2 \
           - 2*corr*(Om-Om_p)*(H0-H0_p)/(sig_Om*sig_H0)
    return chi2

# ========== 总 χ² ==========
def chi2_total(theta, z, mu, cov, bao, is_lcd=False):
    # 只支持振荡模型
    A, b, Om, H0 = theta
    # 边界限制（略宽）
    if not (0 <= A <= 5 and 10 <= b <= 60 and 0.1 <= Om <= 0.5 and 60 <= H0 <= 80):
        return 1e10
    mu_model = np.array([mu_osc(zi, A, b, Om, H0) for zi in z])
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
        D_M, H_z = DM_H_z_osc(z_bao, A, b, Om, H0)
        if not np.isfinite(D_M) or not np.isfinite(H_z):
            return 1e10
        term = (D_M**2) * (C_LIGHT * z_bao / H_z)
        if term <= 0:
            return 1e10
        D_V = term ** (1/3)
        model_val = D_V / RD_FID
        chi2_bao += ((model_val - obs_val) / obs_err)**2
    chi2_planck = planck2018_chi2(Om, H0)
    return chi2_sn + chi2_bao + chi2_planck

# ========== 后验 ==========
def log_prior(theta):
    A, b, Om, H0 = theta
    # 物理先验范围（稍宽于主拟合）
    if 0.5 <= A <= 2.5 and 20 <= b <= 35 and 0.22 <= Om <= 0.32 and 69 <= H0 <= 72:
        return 0.0
    return -np.inf

def log_likelihood(theta, z, mu, cov, bao):
    chi2 = chi2_total(theta, z, mu, cov, bao, is_lcd=False)
    if chi2 > 1e9:
        return -np.inf
    return -0.5 * chi2

def log_posterior(theta, z, mu, cov, bao):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z, mu, cov, bao)

# ========== 主程序 ==========
if __name__ == "__main__":
    print("加载数据...")
    z, mu, cov, bao = load_data()
    print(f"SN点数: {len(z)}")

    # 已知最佳拟合（作为初始位置）
    best_theta = np.array([1.4087, 26.60, 0.2691, 70.16])
    print("最佳拟合:", best_theta)

    # 快速计算该点的 χ² (预热)
    chi2_at_best = chi2_total(best_theta, z, mu, cov, bao, is_lcd=False)
    print(f"χ² at best = {chi2_at_best:.2f}")

    # MCMC 设置
    n_walkers = 16
    n_dim = 4
    n_steps = 2000
    n_burn = 500

    np.random.seed(42)
    pos = best_theta + 1e-4 * np.random.randn(n_walkers, n_dim)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior,
                                        args=(z, mu, cov, bao),
                                        pool=pool)
        print(f"运行 MCMC: {n_steps} 步, {n_walkers} 链, 使用 {mp.cpu_count()} 核...")
        sampler.run_mcmc(pos, n_steps, progress=True)

    samples = sampler.get_chain(discard=n_burn, flat=True)
    print(f"有效样本数: {len(samples)}")

    # 后验统计
    names = ['A', 'b', 'Ωm', 'H0']
    print("\n参数后验中位数及 16%/84% 分位数:")
    for i, name in enumerate(names):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        err_low = q50 - q16
        err_high = q84 - q50
        print(f"{name:4s} = {q50:.4f} +{err_high:.4f} -{err_low:.4f}")

    # 与 Fisher 误差对比（已知）
    fisher_err = [0.0960, 0.4206, 0.0039, 0.1623]
    print("\n对比 Fisher 1σ 误差:")
    for i, name in enumerate(names):
        mcmc_err = (np.percentile(samples[:, i], 84) - np.percentile(samples[:, i], 16)) / 2.0
        print(f"{name:4s} : MCMC = ±{mcmc_err:.4f}, Fisher = ±{fisher_err[i]:.4f}")

    # 保存样本
    np.savetxt("mcmc_samples.txt", samples, header="A b Om H0")
    print("样本保存至 mcmc_samples.txt")

    # 画 corner 图
    try:
        import corner
        fig = corner.corner(samples, labels=names, truths=best_theta,
                            show_titles=True, title_kwargs={"fontsize": 12})
        fig.savefig("corner_mcmc.png", dpi=150)
        print("Corner 图保存至 corner_mcmc.png")
    except ImportError:
        print("未安装 corner，跳过绘图。")