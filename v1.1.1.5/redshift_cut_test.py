import numpy as np
np.random.seed(42)
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import quad
import time
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 从 fit_osc-1.1.1.py 复制的核心参数和函数（未修改）
# ============================================================

C_LIGHT = 299792.458
RD_FID = 147.09
INTEGRAL_EPSREL = 1e-7
INTEGRAL_LIMIT = 500

def load_data(z_min_cut=0.0):
    """加载数据，可剔除红移低于 z_min_cut 的超新星"""
    data = np.genfromtxt("pantheon+_data.txt", skip_header=1, usecols=(4,10,11))
    z_all, mu_all, err_all = data.T
    # 基本掩码：z>0 且有限值
    msk = (z_all > 0) & np.isfinite(mu_all)
    # 红移下限截断
    if z_min_cut > 0:
        msk = msk & (z_all >= z_min_cut)
    z = z_all[msk]
    mu = mu_all[msk]

    with open("Pantheon+SH0ES_STAT+SYS.cov", 'r') as f:
        n = int(f.readline())
    cov_full = np.loadtxt("Pantheon+SH0ES_STAT+SYS.cov", skiprows=1).reshape(n, n)
    cov = cov_full[np.ix_(msk, msk)]
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

def fit_one_cut(z_min_cut):
    """对给定的红移下限，拟合无约束 Ωm≥0.01 的 ΛCDM 和振荡模型"""
    z, mu, cov, bao = load_data(z_min_cut=z_min_cut)
    Om_bounds = (0.01, 0.5)
    opt_options = {'maxiter': 10000, 'finite_diff_rel_step': 1e-5}
    
    # ΛCDM 拟合
    lcdm = minimize(lambda p: chi2_total(p, z, mu, cov, bao, True),
                    [0.3, 70.0],
                    bounds=[Om_bounds, (60,85)],
                    method='L-BFGS-B', options=opt_options)
    Om_l, H0_l = lcdm.x
    chi2_l = lcdm.fun
    
    # 振荡模型拟合
    osc = minimize(lambda p: chi2_total(p, z, mu, cov, bao, False),
                   [0.0, 0.0, 20.0, 0.3, 70.0],
                   bounds=[(0,1.2), (0,30), (0,50), Om_bounds, (60,85)],
                   method='L-BFGS-B', options=opt_options)
    A, a, b, Om_o, H0_o = osc.x
    chi2_o = osc.fun
    
    return {
        'z_min_cut': z_min_cut,
        'N_sne': len(z),
        'Om_l': Om_l, 'H0_l': H0_l, 'chi2_l': chi2_l,
        'A': A, 'alpha': a, 'beta': b,
        'Om_o': Om_o, 'H0_o': H0_o, 'chi2_o': chi2_o,
        'delta': chi2_l - chi2_o
    }

if __name__ == "__main__":
    cut_list = [0.0, 0.05, 0.1, 0.2]
    print("="*80)
    print("红移截断稳健性测试 (固定 r_d=147.09, 无约束 Ωm≥0.01)")
    print("="*80)
    
    results = []
    for zcut in cut_list:
        print(f"\n>>> 正在运行 z_min_cut = {zcut} ...")
        start = time.time()
        res = fit_one_cut(zcut)
        elapsed = time.time() - start
        print(f"    完成，耗时 {elapsed:.1f} 秒")
        print(f"    剩余超新星数量: {res['N_sne']}")
        print(f"    ΛCDM : Ωm={res['Om_l']:.4f}, H0={res['H0_l']:.2f}, χ²={res['chi2_l']:.2f}")
        print(f"    振荡 : A={res['A']:.4f}, α={res['alpha']:.4f}, β={res['beta']:.2f}, Ωm={res['Om_o']:.4f}, H0={res['H0_o']:.2f}, χ²={res['chi2_o']:.2f}")
        print(f"    Δχ² = {res['delta']:.2f}")
        results.append(res)
    
    print("\n" + "="*80)
    print("汇总表格")
    print("="*80)
    print(f"{'z_min':<6} {'N_SNe':<6} {'β':<8} {'A':<8} {'Ωm_o':<8} {'H0_o':<8} {'Δχ²':<8}")
    for r in results:
        print(f"{r['z_min_cut']:<6.2f} {r['N_sne']:<6} {r['beta']:<8.2f} {r['A']:<8.4f} {r['Om_o']:<8.4f} {r['H0_o']:<8.2f} {r['delta']:<8.2f}")