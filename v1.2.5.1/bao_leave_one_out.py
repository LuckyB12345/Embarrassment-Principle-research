import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
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
    bao_full = {0.50: (18.65, 0.25), 0.70: (24.30, 0.30), 1.00: (31.80, 0.45)}
    return z, mu, cov, bao_full

# ========== ΛCDM ==========
def E_lcd(z, Om):
    return np.sqrt(Om*(1+z)**3 + (1-Om))

def mu_lcd(z, Om, H0):
    def integrand(x): return 1.0 / E_lcd(x, Om)
    I, _ = quad(integrand, 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    dc = (C_LIGHT / H0) * I
    return 5 * np.log10(dc * (1+z)) + 25

def DM_H_z_lcd(z, Om, H0):
    def integrand(x): return 1.0 / E_lcd(x, Om)
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

def planck2018_chi2(Om, H0):
    Om_p, H0_p = 0.3111, 67.66
    s_Om, s_H0 = 0.0056, 0.42
    corr = 0.18
    return ((Om-Om_p)/s_Om)**2 + ((H0-H0_p)/s_H0)**2 - 2*corr*(Om-Om_p)*(H0-H0_p)/(s_Om*s_H0)

def chi2_total(theta, z, mu, cov, bao):
    A, b, Om, H0 = theta
    # 物理约束
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

def leave_one_out_bao(z, mu, cov, bao_full):
    z_bao_list = sorted(bao_full.keys())
    results = {}
    for leave_z in z_bao_list:
        print(f"\n=== 去掉 BAO 点 z = {leave_z} ===")
        bao_rest = {z_: bao_full[z_] for z_ in z_bao_list if z_ != leave_z}
        init = [1.4087, 26.60, 0.2691, 70.16]
        bounds = [(0, 5), (10, 60), (0.1, 0.5), (60, 80)]
        res = minimize(lambda p: chi2_total(p, z, mu, cov, bao_rest),
                       init, bounds=bounds, method='L-BFGS-B', options={'maxiter':50000})
        if not res.success:
            print(f"  优化失败: {res.message}")
            continue
        A, b, Om, H0 = res.x
        chi2 = res.fun
        print(f"  拟合结果: A={A:.4f}, b={b:.2f}, Om={Om:.4f}, H0={H0:.2f}, χ²={chi2:.2f}")
        D_M, H_z = DM_H_z_osc(leave_z, A, b, Om, H0)
        term = (D_M**2) * (C_LIGHT * leave_z / H_z)
        if term <= 0:
            print("  预言失败: term <= 0")
            continue
        D_V = term ** (1/3)
        model_val = D_V / RD_FID
        obs_val, obs_err = bao_full[leave_z]
        diff = model_val - obs_val
        sigma = diff / obs_err
        print(f"  预言 D_V/r_d = {model_val:.3f}, 观测 = {obs_val:.3f} ± {obs_err:.3f}")
        print(f"  偏差 = {diff:.3f} ({sigma:.2f}σ)")
        results[leave_z] = (model_val, obs_val, obs_err, sigma)
    return results

if __name__ == "__main__":
    print("加载数据...")
    z, mu, cov, bao_full = load_data()
    print(f"SN 点数: {len(z)}")
    print(f"BAO 红移点: {sorted(bao_full.keys())}")
    results = leave_one_out_bao(z, mu, cov, bao_full)
    print("\n" + "="*50)
    print("留一法总结:")
    for z_bao, (model, obs, err, sigma) in results.items():
        print(f"z={z_bao}: 预言 {model:.3f} vs 观测 {obs:.3f}±{err:.3f} → {sigma:.2f}σ")
        # 验证 BAO 预测
    print("\n=== 验证 BAO 预测 (使用主拟合参数) ===")
    for z_bao in [0.5, 0.7, 1.0]:
        D_M, H_z = DM_H_z_osc(z_bao, A_best, b_best, Om_best, H0_best)
        term = (D_M**2) * (C_LIGHT * z_bao / H_z)
        D_V = term ** (1/3)
        pred = D_V / RD_FID
        obs, err = bao[z_bao]
        print(f"z={z_bao}: pred={pred:.3f}, obs={obs:.3f}±{err:.3f}, diff={pred-obs:.3f} ({abs(pred-obs)/err:.2f}σ)")
        # 同时也打印出 BAO χ² 贡献（从拟合结果中获取）
        chi2_bao_point = ((pred - obs)/err)**2
        print(f"   χ² contribution = {chi2_bao_point:.2f}")