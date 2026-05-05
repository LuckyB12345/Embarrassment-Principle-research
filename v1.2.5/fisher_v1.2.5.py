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
    DM = dc
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
    val = np.clip(val, -50, 50)
    return np.exp(val)

def mu_osc(z, A, b, Om, H0):
    def integrand(x):
        rho_x = rho_osc_at_z(x, A, b)
        rho_x = np.clip(rho_x, 1e-30, 1e30)
        return 1.0 / np.sqrt(Om*(1+x)**3 + (1-Om)*rho_x)
    I, _ = quad(integrand, 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    dc = (C_LIGHT / H0) * I
    return 5 * np.log10(dc * (1+z)) + 25

def DM_H_z_osc(z, A, b, Om, H0):
    def integrand(x):
        rho_x = rho_osc_at_z(x, A, b)
        rho_x = np.clip(rho_x, 1e-30, 1e30)
        return 1.0 / np.sqrt(Om*(1+x)**3 + (1-Om)*rho_x)
    I, _ = quad(integrand, 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    dc = (C_LIGHT / H0) * I
    DM = dc
    rho_z = rho_osc_at_z(z, A, b)
    rho_z = np.clip(rho_z, 1e-30, 1e30)
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
        # 物理约束：z=0.2 处必须满足加速膨胀 (w < -1/3 => 1+w < 2/3)
        if one_plus_w(0.2, A, b) >= 2.0/3.0:
            return 1e10
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
    print("="*70)
    print("物理约束版 v8.0 【尴尬模型 + 普朗克 2018 联合拟合】")
    print("✅ 强制 z=0.2 处 1+w < 2/3 (加速膨胀)")
    print("="*70)

    z, mu, cov, bao = load_data()
    print(f"SN 数据点: {len(z)}")

    print("\n>>> 拟合 ΛCDM ...")
    lcd_res = minimize(lambda p: chi2_total(p, z, mu, cov, bao, True),
                       [0.31, 67.7], bounds=[(0.15, 0.40), (65,78)],
                       method='L-BFGS-B', options={'maxiter':20000})
    chi2_lcd = lcd_res.fun
    Om_l, H0_l = lcd_res.x

    print(">>> 拟合振荡模型 (带加速约束) ...")
    bounds_osc = [(0, 10.0), (10, 50), (0.15, 0.40), (65, 78)]
    init_osc = [0.8, 18.5, 0.30, 69.0]
    osc_res = minimize(lambda p: chi2_total(p, z, mu, cov, bao, False),
                       init_osc, bounds=bounds_osc,
                       method='L-BFGS-B', options={'maxiter':50000})
    A_best, b_best, Om_best, H0_best = osc_res.x
    chi2_osc = osc_res.fun
    delta_chi2 = chi2_lcd - chi2_osc

    print("\n" + "="*50)
    print("ΛCDM 最佳拟合:")
    print(f"  Ωm = {Om_l:.4f}, H0 = {H0_l:.2f}, χ² = {chi2_lcd:.2f}")
    print("振荡模型最佳拟合 (带加速约束):")
    print(f"  A = {A_best:.4f}, b = {b_best:.2f}, Ωm = {Om_best:.4f}, H0 = {H0_best:.2f}, χ² = {chi2_osc:.2f}")
    print(f"Δχ² = {delta_chi2:.2f} (相对于 ΛCDM)")

    # ---------- Fisher 矩阵 ----------
    def chi2_wrap(theta):
        return chi2_total(theta, z, mu, cov, bao, is_lcd=False)

    best = np.array([A_best, b_best, Om_best, H0_best])
    print("\n>>> 计算 Hessian 矩阵 (Fisher) ...")
    H_fish = num_hessian(chi2_wrap, best, eps=1e-4)
    cov_fish = inv(H_fish)
    err_fish = np.sqrt(np.diag(cov_fish))
    names = ['A', 'b', 'Ωm', 'H0']
    print("\nFisher 1σ 误差:")
    for n, e in zip(names, err_fish):
        print(f"  {n:5s} = ±{e:.4f}")
    corr = cov_fish / np.outer(err_fish, err_fish)
    print("\n相关系数矩阵:")
    print(corr)

    z_check = [0.2, 0.4, 0.6, 0.8, 1.0]
    wp1 = [one_plus_w(zi, A_best, b_best) for zi in z_check]
    print("\n📌 1+w(z) 物理检验:")
    for zi, val in zip(z_check, wp1):
        print(f"z={zi:.1f} | 1+w = {val:.4f}")

    print("\n所有分析完成。")