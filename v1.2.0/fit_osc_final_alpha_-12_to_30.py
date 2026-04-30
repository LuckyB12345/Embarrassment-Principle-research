import numpy as np
np.random.seed(42)
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import quad
import time
import warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore")

# ============================================================
# 高精度并行版，α 边界改为 [-12, 30]
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

def chi2_total(theta, z, mu, cov, bao, is_lcdm):
    if is_lcdm:
        Om, H0 = theta
        mu_model = np.array([mu_lcdm(zi, Om, H0) for zi in z])
    else:
        A, a, b, Om, H0 = theta
        # 关键修改：α 边界改为 [-12, 30]
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

def fit_one_constraint(constraint):
    z, mu, cov, bao = load_data()
    Om_bounds = constraint['Om_bounds']
    fix_Om = constraint['fix_Om']
    Om_fixed = constraint.get('Om_fixed', None)
    opt_options = {'maxiter': 10000, 'finite_diff_rel_step': 1e-5}
    if fix_Om:
        # ΛCDM 拟合
        lcdm = minimize(lambda p: chi2_total(p, z, mu, cov, bao, True),
                        [Om_fixed, 70.0],
                        bounds=[(Om_fixed, Om_fixed), (60,85)],
                        method='L-BFGS-B', options=opt_options)
        Om_l, H0_l = lcdm.x
        chi2_l = lcdm.fun
        # 振荡模型拟合（α 边界 -12～30）
        def obj_osc(p):
            A, a, b, H0 = p
            return chi2_total([A, a, b, Om_fixed, H0], z, mu, cov, bao, False)
        osc = minimize(obj_osc, [0.0, -5.0, 18.0, 70.0],
                       bounds=[(0,1.2), (-12,30), (0,50), (60,85)],
                       method='L-BFGS-B', options=opt_options)
        A, a, b, H0_o = osc.x
        Om_o = Om_fixed
        chi2_o = osc.fun
    else:
        # ΛCDM 拟合
        lcdm = minimize(lambda p: chi2_total(p, z, mu, cov, bao, True),
                        [0.3, 70.0],
                        bounds=[Om_bounds, (60,85)],
                        method='L-BFGS-B', options=opt_options)
        Om_l, H0_l = lcdm.x
        chi2_l = lcdm.fun
        # 振荡模型拟合（α 边界 -12～30）
        osc = minimize(lambda p: chi2_total(p, z, mu, cov, bao, False),
                       [0.0, -5.0, 18.0, 0.3, 70.0],
                       bounds=[(0,1.2), (-12,30), (0,50), Om_bounds, (60,85)],
                       method='L-BFGS-B', options=opt_options)
        A, a, b, Om_o, H0_o = osc.x
        chi2_o = osc.fun
    return {
        "constraint": constraint['name'],
        "Om_l": Om_l, "H0_l": H0_l, "chi2_l": chi2_l,
        "A": A, "alpha": a, "beta": b,
        "Om_o": Om_o, "H0_o": H0_o, "chi2_o": chi2_o,
        "delta": chi2_l - chi2_o
    }

if __name__ == "__main__":
    constraints = [
        {"name": "无约束 (Ωm ≥ 0.01)",   "Om_bounds": (0.01, 0.5), "fix_Om": False},
        {"name": "Ωm ≥ 0.2",           "Om_bounds": (0.2, 0.5),   "fix_Om": False},
        {"name": "Ωm ≥ 0.25",          "Om_bounds": (0.25, 0.5),  "fix_Om": False},
        {"name": "Ωm 固定 = 0.315",     "Om_bounds": None,         "fix_Om": True,  "Om_fixed": 0.315}
    ]

    print("\n" + "="*80)
    print("主拟合 (α 边界 -12～30)")
    print("="*80)

    start_time = time.time()
    n_cores = min(4, 4)
    with Pool(processes=n_cores) as pool:
        results = pool.map(fit_one_constraint, constraints)
    elapsed = time.time() - start_time
    print(f"\n⏱️ 总耗时: {elapsed:.1f} 秒")
    print("="*80)

    print("\n" + "="*80)
    print("汇总表格")
    print("="*80)
    print("{:<20} {:<6} {:<6} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
        "约束", "Ωm_L", "H0_L", "χ²_L", "A", "α", "β", "Ωm_O", "H0_O", "χ²_O", "Δχ²"))
    for r in results:
        print("{:<20} {:<6.4f} {:<6.2f} {:<8.2f} {:<8.4f} {:<8.4f} {:<8.2f} {:<8.4f} {:<8.2f} {:<8.2f}".format(
            r['constraint'], r['Om_l'], r['H0_l'], r['chi2_l'],
            r['A'], r['alpha'], r['beta'], r['Om_o'], r['H0_o'], r['delta']))