"""
MCMC 分析：DESI DR2 6点 BAO + Pantheon+ SN + Planck 2018
模型：1+w(z) = A sin(bz)/(1+z)^2
BAO 数据：星系/类星体部分 (z=0.20,0.51,0.70,0.93,1.32,1.48)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from astropy.cosmology import FLRW
from astropy import units as u
from scipy.integrate import quad
import emcee
import corner
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

# ========== 常量 ==========
C_LIGHT = 299792.458      # km/s
RD_FID = 147.09           # Mpc

# ========== 数据加载 ==========
def load_data():
    # Pantheon+ SN
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

    # DESI DR2 6点 (不含 Lyα)
    bao = {
        0.20: (3.861, 0.045),
        0.51: (13.767, 0.061),
        0.70: (18.651, 0.071),
        0.93: (22.616, 0.068),
        1.32: (27.660, 0.110),
        1.48: (29.550, 0.150),
    }
    return z, mu, cov, bao

# ========== 振荡模型 ==========
class OscillatingDE(FLRW):
    def __init__(self, H0, Om0, A, b, name="OscillatingDE"):
        self._A = A
        self._b = b
        super().__init__(H0=H0*u.km/u.s/u.Mpc, Om0=Om0, Ode0=1-Om0,
                         Tcmb0=2.7255*u.K, Neff=3.046, name=name)
    def w(self, z):
        return -1.0 + self._A * np.sin(self._b * z) / (1+z)**2

def mu_cosmo(cosmo, z):
    d_L = cosmo.luminosity_distance(z).value
    return 5 * np.log10(d_L) + 25

def bao_D_V(cosmo, z):
    d_c = cosmo.comoving_distance(z).value
    H_z = cosmo.H(z).value
    term = (d_c**2) * (C_LIGHT * z / H_z)
    D_V = term ** (1/3)
    return D_V / RD_FID

def planck2018_chi2(Om, H0):
    Om_p, H0_p = 0.3111, 67.66
    s_Om, s_H0 = 0.0056, 0.42
    corr = 0.18
    return ((Om-Om_p)/s_Om)**2 + ((H0-H0_p)/s_H0)**2 - 2*corr*(Om-Om_p)*(H0-H0_p)/(s_Om*s_H0)

def chi2_total(params, z_sn, mu_sn, cov_sn, bao_data):
    A, b, Om, H0 = params
    # 物理约束：z=0.2 处加速膨胀 (1+w < 2/3)
    if -1.0 + A * np.sin(b*0.2)/(1.2)**2 >= -1/3:
        return 1e10
    if not (0 <= A <= 10 and 10 <= b <= 50 and 0.01 <= Om <= 0.50 and 65 <= H0 <= 78):
        return 1e10
    cosmo = OscillatingDE(H0=H0, Om0=Om, A=A, b=b)
    mu_model = np.array([mu_cosmo(cosmo, zi) for zi in z_sn])
    if not np.all(np.isfinite(mu_model)):
        return 1e10
    resid = mu_sn - mu_model
    try:
        cho = cho_factor(cov_sn)
        chi2_sn = resid @ cho_solve(cho, resid)
    except:
        return 1e10
    chi2_bao = 0.0
    for z_bao, (obs, err) in bao_data.items():
        pred = bao_D_V(cosmo, z_bao)
        chi2_bao += ((pred - obs) / err)**2
    chi2_planck = planck2018_chi2(Om, H0)
    return chi2_sn + chi2_bao + chi2_planck

# ========== MCMC 后验 ==========
def log_prior(theta):
    A, b, Om, H0 = theta
    if 0 <= A <= 10 and 10 <= b <= 50 and 0.01 <= Om <= 0.50 and 65 <= H0 <= 78:
        return 0.0
    return -np.inf

def log_likelihood(theta, z_sn, mu_sn, cov_sn, bao_data):
    chi2 = chi2_total(theta, z_sn, mu_sn, cov_sn, bao_data)
    if chi2 > 1e9:
        return -np.inf
    return -0.5 * chi2

def log_posterior(theta, z_sn, mu_sn, cov_sn, bao_data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z_sn, mu_sn, cov_sn, bao_data)

# ========== 主程序 ==========
if __name__ == "__main__":
    print("="*70)
    print("MCMC: DESI DR2 6点 BAO + Pantheon+ + Planck 2018")
    print("振荡暗能量模型: 1+w(z) = A sin(bz)/(1+z)^2")
    print("="*70)

    z_sn, mu_sn, cov_sn, bao_data = load_data()
    print(f"SN 数据点: {len(z_sn)}")
    print(f"BAO 数据点: {sorted(bao_data.keys())}")

    # 先用优化找到最佳拟合作为初始位置
    init = [1.5403, 26.42, 0.1252, 69.89]   # 基于6点联合拟合的最佳值
    bounds = [(0,10), (10,50), (0.01,0.50), (65,78)]
    res = minimize(lambda p: chi2_total(p, z_sn, mu_sn, cov_sn, bao_data),
                   init, bounds=bounds, method='L-BFGS-B', options={'maxiter':50000})
    best = res.x
    print("优化最佳拟合:", best)
    print("χ² at best:", res.fun)

    # MCMC 设置
    n_walkers = 20
    n_dim = 4
    n_steps = 3000
    n_burn = 1000

    np.random.seed(42)
    pos = best + 1e-4 * np.random.randn(n_walkers, n_dim)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior,
                                        args=(z_sn, mu_sn, cov_sn, bao_data),
                                        pool=pool)
        print(f"运行 MCMC: {n_steps} 步, {n_walkers} 链, 使用 {mp.cpu_count()} 核...")
        sampler.run_mcmc(pos, n_steps, progress=True)

    samples = sampler.get_chain(discard=n_burn, flat=True)
    print(f"有效样本数: {len(samples)}")

    names = ['A', 'b', 'Ωm', 'H0']
    print("\n参数后验中位数及 16%/84% 分位数:")
    for i, name in enumerate(names):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{name:4s} = {q50:.4f} +{q84-q50:.4f} -{q50-q16:.4f}")

    # 保存样本和 corner 图
    np.savetxt("mcmc_samples_6points.txt", samples, header="A b Om H0")
    try:
        fig = corner.corner(samples, labels=names, truths=best,
                            show_titles=True, title_kwargs={"fontsize": 12})
        fig.savefig("corner_mcmc_6points.png", dpi=150)
        print("Corner 图保存至 corner_mcmc_6points.png")
    except ImportError:
        print("未安装 corner，跳过绘图。")

    print("\n所有分析完成。")