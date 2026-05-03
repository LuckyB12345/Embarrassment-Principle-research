import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")

# ========== 常量 ==========
C_LIGHT = 299792.458
RD_FID = 147.09
INTEGRAL_LIMIT = 500

# 全局精度变量（将会被动态修改）
INTEGRAL_EPSREL = 1e-5   # 默认值，后面会改

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

# ========== ΛCDM 函数 ==========
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
def chi2_total(theta, z, mu, cov, bao, is_lcd):
    if is_lcd:
        Om, H0 = theta
        mu_model = np.array([mu_lcd(zi, Om, H0) for zi in z])
    else:
        A, b, Om, H0 = theta
        # 边界放宽到10，但只检验两个点，不影响
        if not (0 <= A <= 10.0 and 10 <= b <= 50 and 0.15 <= Om <= 0.40 and 65 <= H0 <= 78):
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

    chi2_planck = planck2018_chi2(Om, H0)
    return chi2_sn + chi2_bao + chi2_planck

# ========== 高精度封装 ==========
def compute_chi2(theta, eps_rel, z, mu, cov, bao):
    global INTEGRAL_EPSREL
    original = INTEGRAL_EPSREL
    INTEGRAL_EPSREL = eps_rel
    chi2 = chi2_total(theta, z, mu, cov, bao, is_lcd=False)
    INTEGRAL_EPSREL = original
    return chi2

# ========== 主程序 ==========
if __name__ == "__main__":
    print("加载数据...")
    z, mu, cov, bao = load_data()
    print(f"SN 点数: {len(z)}")

    candidate = [2.3154, 37.56, 0.2683, 69.57]
    best = [1.4087, 26.60, 0.2691, 70.16]

    # 先用1e-7验证（已有的结果快速过一遍）
    print("\n精度 1e-7 下计算...")
    chi2_cand_7 = compute_chi2(candidate, 1e-7, z, mu, cov, bao)
    chi2_best_7 = compute_chi2(best, 1e-7, z, mu, cov, bao)
    print(f"候选点 χ² = {chi2_cand_7:.2f}")
    print(f"主拟合点 χ² = {chi2_best_7:.2f}")
    print(f"差值 = {chi2_cand_7 - chi2_best_7:.2f}")

    # 再用 1e-8 超高精度（可能较慢，但只两个点）
    print("\n精度 1e-8 下计算（可能需要几分钟到半小时）...")
    chi2_cand_8 = compute_chi2(candidate, 1e-8, z, mu, cov, bao)
    chi2_best_8 = compute_chi2(best, 1e-8, z, mu, cov, bao)
    print(f"候选点 χ² = {chi2_cand_8:.2f}")
    print(f"主拟合点 χ² = {chi2_best_8:.2f}")
    print(f"差值 = {chi2_cand_8 - chi2_best_8:.2f}")

    print("\n高精度验证完成。")