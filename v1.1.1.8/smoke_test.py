import numpy as np
np.random.seed(42)
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 以下是从 fit_osc-1.1.1.py 复制的核心函数（仅用于烟雾测试）
# ============================================================

C_LIGHT = 299792.458
RD_FID = 147.09
INTEGRAL_EPSREL = 1e-7
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

def fit_one(z, mu, cov, bao, Om_bounds, fix_Om=False, Om_fixed=None):
    opt_options = {'maxiter': 10000, 'finite_diff_rel_step': 1e-5}
    if fix_Om:
        lcdm = minimize(lambda p: chi2_total(p, z, mu, cov, bao, True),
                        [Om_fixed, 70.0],
                        bounds=[(Om_fixed, Om_fixed), (60,85)],
                        method='L-BFGS-B', options=opt_options)
        Om_l, H0_l = lcdm.x
        chi2_l = lcdm.fun
        def obj_osc(p):
            A, a, b, H0 = p
            return chi2_total([A, a, b, Om_fixed, H0], z, mu, cov, bao, False)
        osc = minimize(obj_osc, [0.0, 0.0, 20.0, 70.0],
                       bounds=[(0,1.2), (0,30), (0,50), (60,85)],
                       method='L-BFGS-B', options=opt_options)
        A, a, b, H0_o = osc.x
        Om_o = Om_fixed
        chi2_o = osc.fun
    else:
        lcdm = minimize(lambda p: chi2_total(p, z, mu, cov, bao, True),
                        [0.3, 70.0],
                        bounds=[Om_bounds, (60,85)],
                        method='L-BFGS-B', options=opt_options)
        Om_l, H0_l = lcdm.x
        chi2_l = lcdm.fun
        osc = minimize(lambda p: chi2_total(p, z, mu, cov, bao, False),
                       [0.0, 0.0, 20.0, 0.3, 70.0],
                       bounds=[(0,1.2), (0,30), (0,50), Om_bounds, (60,85)],
                       method='L-BFGS-B', options=opt_options)
        A, a, b, Om_o, H0_o = osc.x
        chi2_o = osc.fun
    return Om_l, H0_l, chi2_l, A, a, b, Om_o, H0_o, chi2_o

# ============================================================
# 烟雾测试主程序
# ============================================================

def generate_mock_data(z, mu_lcdm_pred, cov):
    L = cholesky(cov, lower=True)
    noise = L @ np.random.randn(len(z))
    return mu_lcdm_pred + noise

def main():
    z, mu_real, cov, bao = load_data()
    Om_true = 0.0177
    H0_true = 76.25
    mu_lcdm_true = np.array([mu_lcdm(zi, Om_true, H0_true) for zi in z])
    mu_sim = generate_mock_data(z, mu_lcdm_true, cov)
    Om_bounds = (0.01, 0.5)
    Om_l, H0_l, chi2_l, A, a, b, Om_o, H0_o, chi2_o = fit_one(
        z, mu_sim, cov, bao, Om_bounds, fix_Om=False)
    delta = chi2_l - chi2_o
    print("="*60)
    print("烟雾测试：用ΛCDM模拟数据（无振荡）检验代码")
    print(f"模拟数据基于: Ωm={Om_true}, H0={H0_true}")
    print(f"ΛCDM拟合结果: Ωm={Om_l:.4f}, H0={H0_l:.2f}, χ²={chi2_l:.2f}")
    print(f"振荡模型拟合: A={A:.4f}, α={a:.4f}, β={b:.2f}, Ωm={Om_o:.4f}, H0={H0_o:.2f}, χ²={chi2_o:.2f}")
    print(f"Δχ² = {delta:.2f} (预期接近0)")
    if abs(delta) < 5:
        print("✅ 测试通过：代码没有系统性偏差")
    else:
        print("⚠️ 测试失败：Δχ²异常偏大，请检查代码")
    print("="*60)

if __name__ == "__main__":
    main()