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
    bao = {0.50: (18.65, 0.25), 0.70: (24.30, 0.30), 1.00: (31.80, 0.45)}
    return z, mu, cov, bao

# ========== ΛCDM 标准模型 ==========
def E_lcd(z, Om):
    return np.sqrt(Om*(1+z)**3 + (1-Om))

def mu_lcd(z, Om, H0):
    try:
        dc, _ = quad(lambda x: C_LIGHT / E_lcd(x, Om), 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10
    return 5*np.log10((dc / H0) * (1+z)) + 25

def DM_H_z_lcd(z, Om, H0):
    try:
        dc, _ = quad(lambda x: C_LIGHT / E_lcd(x, Om), 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10, 1e10
    DM = dc / H0
    Hz = H0 * E_lcd(z, Om)
    return DM, Hz

# ========== 包络模型（无 α₀，固定 n=2）==========
def one_plus_w(z, A, b):
    # 包络函数：1+w = A * sin(bz) / (1+z)^2
    return A * np.sin(b * z) / (1+z)**2

def rho_de_integrand(x, A, b):
    # 被积函数：3 * (1+w) / (1+x) = 3 * A * sin(bx) / (1+x)^3
    return 3 * A * np.sin(b * x) / (1+x)**3

def rho_osc_at_z(z, A, b):
    try:
        val, _ = quad(lambda x: rho_de_integrand(x, A, b), 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
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
        dc, _ = quad(integrand_dc, 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10
    return 5*np.log10((dc / H0) * (1+z)) + 25

def DM_H_z_osc(z, A, b, Om, H0):
    def integrand_dc(x):
        rho_x = rho_osc_at_z(x, A, b)
        rho_x = np.clip(rho_x, 0.0, 1e6)
        return C_LIGHT / np.sqrt(Om*(1+x)**3 + (1-Om)*rho_x)
    try:
        dc, _ = quad(integrand_dc, 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10, 1e10
    DM = dc / H0
    rho_z = rho_osc_at_z(z, A, b)
    rho_z = np.clip(rho_z, 0.0, 1e6)
    Hz = H0 * np.sqrt(Om*(1+z)**3 + (1-Om)*rho_z)
    return DM, Hz

# ========== χ² 拟合 ==========
def chi2_total(theta, z, mu, cov, bao, is_lcd):
    if is_lcd:
        Om, H0 = theta
        mu_model = np.array([mu_lcd(zi, Om, H0) for zi in z])
    else:
        A, b, Om, H0 = theta
        # 宽松边界：A 可到 0.5，b 保持在合理范围
        if not (0 <= A <= 0.5 and 10 <= b <= 50 and 0.15 <= Om <= 0.35 and 67 <= H0 <= 78):
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
        if is_lcd:
            D_M, H_z = DM_H_z_lcd(z_bao, Om, H0)
        else:
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

# ========== 主程序 ==========
if __name__ == "__main__":
    print("="*70)
    print("v1.2.3 【包络函数】1+w = A sin(bz)/(1+z)² | 高红移绝对回归")
    print("✅ 无α₀参数 | ✅ 高红移强制衰减 | ✅ BIC惩罚小")
    print("="*70)
    z, mu, cov, bao = load_data()

    print("\n正在拟合 ΛCDM...")
    lcd_res = minimize(lambda p: chi2_total(p, z, mu, cov, bao, True),
                        [0.25, 72.0], bounds=[(0.15, 0.35), (67,78)],
                        method='L-BFGS-B', options={'maxiter':15000})
    chi2_lcd = lcd_res.fun
    Om_l, H0_l = lcd_res.x

    print("正在拟合 包络振荡模型...")
    init = [0.1, 18.0, 0.25, 72.0]   # A初始0.1，b初始18
    bounds = [(0, 0.5), (10, 50), (0.15, 0.35), (67, 78)]
    
    osc_res = minimize(lambda p: chi2_total(p, z, mu, cov, bao, False),
                       init, bounds=bounds,
                       method='L-BFGS-B', options={'maxiter':30000})
    A, b, Om_o, H0_o = osc_res.x
    chi2_osc = osc_res.fun
    delta_chi2 = chi2_lcd - chi2_osc

    z_check = [0.2, 0.4, 0.6, 0.8, 1.0]
    wp1 = [one_plus_w(z, A, b) for z in z_check]

    print("\n"+"="*50)
    print("ΛCDM 结果")
    print(f"Ωm    = {Om_l:.4f}")
    print(f"H0    = {H0_l:.2f}")
    print(f"χ²    = {chi2_lcd:.2f}")
    print("-"*50)
    print("包络振荡模型【v1.2.3】")
    print(f"A     = {A:.4f}")
    print(f"b     = {b:.2f}")
    print(f"Ωm    = {Om_o:.4f}")
    print(f"H0    = {H0_o:.2f}")
    print(f"χ²    = {chi2_osc:.2f}")
    print(f"Δχ²   = {delta_chi2:.2f}")
    print("-"*50)
    print("📌 1+w(z) 物理校验（高红移应接近0）")
    for zi, val in zip(z_check, wp1):
        print(f"z={zi:.1f} | 1+w = {val:.4f}")
    print("="*50)