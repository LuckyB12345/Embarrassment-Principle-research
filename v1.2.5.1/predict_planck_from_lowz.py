import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, inv
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")

# ========== 常量 ==========
C_LIGHT = 299792.458
RD_FID = 147.09
INTEGRAL_EPSREL = 1e-7
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

# ========== ΛCDM ==========
def E_lcd(z, Om):
    return np.sqrt(Om*(1+z)**3 + (1-Om))

def mu_lcd(z, Om, H0):
    def integrand(x):
        return 1.0 / E_lcd(x, Om)
    I, _ = quad(integrand, 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    dc = (C_LIGHT / H0) * I
    return 5 * np.log10(dc * (1+z)) + 25

def DM_H_z_lcd(z, Om, H0):
    def integrand(x):
        return 1.0 / E_lcd(x, Om)
    I, _ = quad(integrand, 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    dc = (C_LIGHT / H0) * I
    return dc, H0 * E_lcd(z, Om)

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

# ========== Planck 压缩似然（此处不用，但保留）==========
def planck2018_chi2(Om, H0):
    Om_p, H0_p = 0.3111, 67.66
    s_Om, s_H0 = 0.0056, 0.42
    corr = 0.18
    return ((Om-Om_p)/s_Om)**2 + ((H0-H0_p)/s_H0)**2 - 2*corr*(Om-Om_p)*(H0-H0_p)/(s_Om*s_H0)

# ========== 总 χ²（不含 Planck 先验，但含物理约束）==========
def chi2_total_no_planck(theta, z, mu, cov, bao):
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
    return chi2_sn + chi2_bao

# ========== Fisher 矩阵 ==========
def num_hessian(f, x, eps=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        H[i,i] = (f(xp) - 2*f0 + f(xm)) / (eps**2)
        for j in range(i+1, n):
            xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i,j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps**2)
            H[j,i] = H[i,j]
    return H

# ========== 主程序 ==========
if __name__ == "__main__":
    z, mu, cov, bao = load_data()
    print("只用 SN+BAO 拟合振荡模型（无 Planck 先验，含物理约束）...")

    init = [1.4087, 26.60, 0.2691, 70.16]
    bounds = [(0, 5), (10, 60), (0.1, 0.5), (60, 80)]
    res = minimize(lambda p: chi2_total_no_planck(p, z, mu, cov, bao),
                   init, bounds=bounds, method='L-BFGS-B', options={'maxiter':50000})
    A_best, b_best, Om_best, H0_best = res.x
    chi2 = res.fun
    print(f"Best-fit: A={A_best:.4f}, b={b_best:.2f}, Om={Om_best:.4f}, H0={H0_best:.2f}, χ²={chi2:.2f}")

    def chi2_wrap(theta):
        return chi2_total_no_planck(theta, z, mu, cov, bao)
    best_array = np.array([A_best, b_best, Om_best, H0_best])
    H_fish = num_hessian(chi2_wrap, best_array, eps=1e-4)
    cov_fish = inv(H_fish)
    err_fish = np.sqrt(np.diag(cov_fish))
    print(f"\nFisher 1σ 误差: A ±{err_fish[0]:.4f}, b ±{err_fish[1]:.4f}, Om ±{err_fish[2]:.4f}, H0 ±{err_fish[3]:.2f}")

    Om_p, H0_p = 0.3111, 67.66
    sig_Om, sig_H0 = 0.0056, 0.42
    print(f"\nPlanck 2018: Om = {Om_p} ± 0.0056, H0 = {H0_p} ± 0.42")
    diff_Om = (Om_best - Om_p) / sig_Om
    diff_H0 = (H0_best - H0_p) / sig_H0
    print(f"\n偏差 (σ): Om = {diff_Om:.2f}σ, H0 = {diff_H0:.2f}σ")
    if abs(diff_Om) < 2 and abs(diff_H0) < 2:
        print("✅ 低红移预言与 Planck 高红移约束在 2σ 内一致 → 模型自洽性非常强。")
    else:
        print("⚠️ 偏差超过 2σ → 模型依赖于 Planck 先验，需要谨慎讨论。")