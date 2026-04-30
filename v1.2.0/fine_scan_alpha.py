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
    cov = np.asarray(cov, dtype=float)
    cov += np.eye(cov.shape[0]) * 1e-8

    bao = {0.50: (18.65, 0.25), 0.70: (24.30, 0.30), 1.00: (31.80, 0.45)}
    return z, mu, cov, bao

# ========== ΛCDM 函数 ==========
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

# ========== 振荡模型 ==========
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

# ========== χ² 计算 ==========
def chi2_total(theta, z, mu, cov, bao, is_lcdm):
    if is_lcdm:
        Om, H0 = theta
        mu_model = np.array([mu_lcdm(zi, Om, H0) for zi in z])
    else:
        A, a, b, Om, H0 = theta
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

# ========== 固定 α 拟合（带递推初始值） ==========
def fit_fixed_alpha(alpha_fixed, z, mu, cov, bao, init_guess=None):
    def obj(theta):
        A, b, Om, H0 = theta
        return chi2_total([A, alpha_fixed, b, Om, H0], z, mu, cov, bao, False)
    if init_guess is None:
        # 默认初始值
        if alpha_fixed < -3:
            ini = [0.3, 17.8, 0.045, 74.5]
        else:
            ini = [0.77, 18.2, 0.01, 73.5]
    else:
        ini = init_guess
    bounds = [(0, 1.2), (10, 50), (0.01, 0.5), (60, 85)]
    res = minimize(obj, ini, bounds=bounds, method='L-BFGS-B', options={'maxiter': 12000})
    if not res.success:
        return np.inf, np.zeros(4)
    return res.fun, res.x

# ========== 主程序 ==========
if __name__ == "__main__":
    print("加载数据...")
    z, mu, cov, bao = load_data()

    # 精细扫描 α = -8.0, -8.5, -9.0, -9.5, -10.0, -10.5, -11.0, -11.5, -12.0
    alpha_fine = np.arange(-8.0, -12.1, -0.5)
    results_fine = []
    prev_params = None
    print("\n===== 精细扫描 (步长 0.5) =====")
    for a in alpha_fine:
        # 使用前一个结果作为初始值，避免跳回 ΛCDM
        if prev_params is not None:
            init_guess = prev_params
        else:
            init_guess = None
        chi2, params = fit_fixed_alpha(a, z, mu, cov, bao, init_guess=init_guess)
        A, b, Om, H0 = params
        # 计算 ρ_de 在 z=1 处检查稳定性
        rho_z1 = None
        if chi2 < 1e9:
            try:
                rho_z1 = rho_osc_at_z(1.0, A, a, b)
            except:
                rho_z1 = -1.0
        results_fine.append((a, chi2, A, b, Om, H0, rho_z1))
        print(f"α = {a:5.2f} | χ² = {chi2:.2f} | A={A:.4f}, β={b:.2f}, Ωm={Om:.4f}, H0={H0:.2f} | ρ_de(z=1)={rho_z1:.2e}")
        if chi2 < 1e9:
            prev_params = params   # 传递下去

    # 自由拟合：α 作为自由参数，边界 [-12, 30]
    print("\n===== 自由拟合 (α ∈ [-12, 30]) =====")
    def obj_free(theta):
        A, a, b, Om, H0 = theta
        return chi2_total([A, a, b, Om, H0], z, mu, cov, bao, False)
    # 初始值取 α=-8 的最佳结果
    best_fixed = min(results_fine, key=lambda x: x[1])  # (a, chi2, A, b, Om, H0, rho)
    init_free = [best_fixed[2], best_fixed[0], best_fixed[3], best_fixed[4], best_fixed[5]]
    bounds_free = [(0, 1.2), (-12, 30), (10, 50), (0.01, 0.5), (60, 85)]
    res_free = minimize(obj_free, init_free, bounds=bounds_free, method='L-BFGS-B', options={'maxiter': 15000})
    if res_free.success:
        A_free, a_free, b_free, Om_free, H0_free = res_free.x
        chi2_free = res_free.fun
        print(f"自由拟合: α = {a_free:.6f}, χ² = {chi2_free:.2f}")
        print(f"A={A_free:.4f}, β={b_free:.2f}, Ωm={Om_free:.4f}, H0={H0_free:.2f}")
    else:
        print("自由拟合失败")
        a_free, chi2_free = np.nan, np.nan

    # 最终推荐
    print("\n" + "="*60)
    print("推荐：")
    if not np.isnan(a_free):
        print(f"采用自由拟合结果: α = {a_free:.2f} (χ² = {chi2_free:.2f})")
    else:
        best = min(results_fine, key=lambda x: x[1])
        print(f"采用精细扫描最优: α = {best[0]:.2f} (χ² = {best[1]:.2f})")
    print("注：α = -12 及更负时出现暗能量密度异常，不予采用。")