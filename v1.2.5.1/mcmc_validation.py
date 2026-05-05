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

C_LIGHT = 299792.458
RD_FID = 147.09
INTEGRAL_EPSREL = 1e-5
INTEGRAL_LIMIT = 500

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

def one_plus_w(z, A, b):
    return A * np.sin(b * z) / (1+z)**2

def rho_osc_at_z(z, A, b):
    def integrand(x): return 3 * one_plus_w(x, A, b) / (1+x)
    val, _ = quad(integrand, 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    val = np.clip(val, -50, 50)
    return np.exp(val)

def mu_osc(z, A, b, Om, H0):
    def integrand(x):
        rho = rho_osc_at_z(x, A, b)
        rho = np.clip(rho, 1e-30, 1e30)
        return 1.0 / np.sqrt(Om*(1+x)**3 + (1-Om)*rho)
    I, _ = quad(integrand, 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    dc = (C_LIGHT / H0) * I
    return 5 * np.log10(dc * (1+z)) + 25

def DM_H_z_osc(z, A, b, Om, H0):
    def integrand(x):
        rho = rho_osc_at_z(x, A, b)
        rho = np.clip(rho, 1e-30, 1e30)
        return 1.0 / np.sqrt(Om*(1+x)**3 + (1-Om)*rho)
    I, _ = quad(integrand, 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    dc = (C_LIGHT / H0) * I
    rho_z = rho_osc_at_z(z, A, b)
    rho_z = np.clip(rho_z, 1e-30, 1e30)
    Hz = H0 * np.sqrt(Om*(1+z)**3 + (1-Om)*rho_z)
    return dc, Hz

def planck2018_chi2(Om, H0):
    Om_p, H0_p = 0.3111, 67.66
    s_Om, s_H0 = 0.0056, 0.42
    corr = 0.18
    return ((Om-Om_p)/s_Om)**2 + ((H0-H0_p)/s_H0)**2 - 2*corr*(Om-Om_p)*(H0-H0_p)/(s_Om*s_H0)

def chi2_total(theta, z, mu, cov, bao):
    A, b, Om, H0 = theta
    # 物理约束：z=0.2 处必须加速膨胀 (1+w < 2/3)
    if one_plus_w(0.2, A, b) >= 2.0/3.0:
        return 1e10
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

def log_prior(theta):
    A, b, Om, H0 = theta
    if 0.5 <= A <= 2.5 and 20 <= b <= 35 and 0.22 <= Om <= 0.32 and 69 <= H0 <= 72:
        return 0.0
    return -np.inf

def log_likelihood(theta, z, mu, cov, bao):
    chi2 = chi2_total(theta, z, mu, cov, bao)
    if chi2 > 1e9:
        return -np.inf
    return -0.5 * chi2

def log_posterior(theta, z, mu, cov, bao):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z, mu, cov, bao)

if __name__ == "__main__":
    z, mu, cov, bao = load_data()
    print(f"SN点数: {len(z)}")
    best_theta = np.array([1.4087, 26.60, 0.2691, 70.16])
    print("最佳拟合:", best_theta)
    chi2_at_best = chi2_total(best_theta, z, mu, cov, bao)
    print(f"χ² at best = {chi2_at_best:.2f}")

    n_walkers, n_dim, n_steps, n_burn = 16, 4, 2000, 500
    np.random.seed(42)
    pos = best_theta + 1e-4 * np.random.randn(n_walkers, n_dim)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior,
                                        args=(z, mu, cov, bao), pool=pool)
        print(f"运行 MCMC: {n_steps} 步, {n_walkers} 链, 使用 {mp.cpu_count()} 核...")
        sampler.run_mcmc(pos, n_steps, progress=True)

    samples = sampler.get_chain(discard=n_burn, flat=True)
    print(f"有效样本数: {len(samples)}")

    names = ['A', 'b', 'Ωm', 'H0']
    print("\n参数后验中位数及 16%/84% 分位数:")
    for i, name in enumerate(names):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{name:4s} = {q50:.4f} +{q84-q50:.4f} -{q50-q16:.4f}")

    fisher_err = [0.0960, 0.4206, 0.0039, 0.1623]
    print("\n对比 Fisher 1σ 误差:")
    for i, name in enumerate(names):
        mcmc_err = (np.percentile(samples[:, i], 84) - np.percentile(samples[:, i], 16)) / 2.0
        print(f"{name:4s} : MCMC = ±{mcmc_err:.4f}, Fisher = ±{fisher_err[i]:.4f}")

    np.savetxt("mcmc_samples.txt", samples, header="A b Om H0")
    try:
        fig = corner.corner(samples, labels=names, truths=best_theta,
                            show_titles=True, title_kwargs={"fontsize": 12})
        fig.savefig("corner_mcmc.png", dpi=150)
        print("Corner 图保存至 corner_mcmc.png")
    except ImportError:
        print("未安装 corner，跳过绘图。")