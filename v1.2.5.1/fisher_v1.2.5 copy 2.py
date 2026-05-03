import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, inv
from scipy.integrate import quad
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ========== 常量 ==========
C_LIGHT = 299792.458
RD_FID = 147.09
INTEGRAL_EPSREL = 1e-5      # 默认精度，快速计算
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

# ========== Fisher 矩阵辅助 ==========
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

# ========== 高精度验证函数 ==========
def verify_chi2(theta, label, z, mu, cov, bao, eps_high=1e-7):
    """用高精度重新计算χ²，并恢复原精度"""
    global INTEGRAL_EPSREL
    original_eps = INTEGRAL_EPSREL
    INTEGRAL_EPSREL = eps_high
    chi2 = chi2_total(theta, z, mu, cov, bao, is_lcd=False)
    INTEGRAL_EPSREL = original_eps
    print(f"{label} χ² (eps={eps_high:.0e}) = {chi2:.2f}")
    return chi2

# ========== 主程序 ==========
if __name__ == "__main__":
    print("=" * 70)
    print("最终分析 v2：拟合 + Fisher + 高精度验证")
    print("默认精度 epsrel =", INTEGRAL_EPSREL)
    print("=" * 70)

    # 加载数据
    z, mu, cov, bao = load_data()
    print(f"SN 数据点: {len(z)}")

    # ---------- 1. 主拟合 ----------
    print("\n>>> 拟合 ΛCDM ...")
    lcd_res = minimize(lambda p: chi2_total(p, z, mu, cov, bao, True),
                       [0.31, 67.7], bounds=[(0.15, 0.40), (65,78)],
                       method='L-BFGS-B', options={'maxiter':20000})
    chi2_lcd = lcd_res.fun
    Om_l, H0_l = lcd_res.x

    print(">>> 拟合振荡模型 ...")
    bounds_osc = [(0, 10.0), (10, 50), (0.15, 0.40), (65, 78)]
    init_osc = [0.8, 18.5, 0.30, 69.0]
    osc_res = minimize(lambda p: chi2_total(p, z, mu, cov, bao, False),
                       init_osc, bounds=bounds_osc,
                       method='L-BFGS-B', options={'maxiter':50000})
    A_best, b_best, Om_best, H0_best = osc_res.x
    chi2_osc = osc_res.fun
    delta_chi2 = chi2_lcd - chi2_osc

    print("\n" + "=" * 50)
    print("ΛCDM 最佳拟合:")
    print(f"  Ωm = {Om_l:.4f}, H0 = {H0_l:.2f}, χ² = {chi2_lcd:.2f}")
    print("振荡模型最佳拟合:")
    print(f"  A = {A_best:.4f}, b = {b_best:.2f}, Ωm = {Om_best:.4f}, H0 = {H0_best:.2f}, χ² = {chi2_osc:.2f}")
    print(f"Δχ² = {delta_chi2:.2f} (相对于 ΛCDM)")

    # ---------- 2. Fisher 矩阵 ----------
    def chi2_wrap(theta):
        return chi2_total(theta, z, mu, cov, bao, is_lcd=False)

    best = np.array([A_best, b_best, Om_best, H0_best])
    print("\n>>> 计算 Hessian 矩阵 (Fisher) ...")
    H_fish = num_hessian(chi2_wrap, best, eps=1e-4)
    cov_fish = inv(H_fish)
    err_fish = np.sqrt(np.diag(cov_fish))
    names = ['A', 'b', 'Ωm', 'H0']
    print("\nFisher 1σ 误差（完整 Hessian）:")
    for n, e in zip(names, err_fish):
        print(f"  {n:5s} = ±{e:.4f}")
    corr = cov_fish / np.outer(err_fish, err_fish)
    print("\n相关系数矩阵:")
    print(corr)

    # ---------- 3. 高精度验证候选点（A≈2.32, b≈37.6）----------
    candidate = [2.3154, 37.56, 0.2683, 69.57]
    print("\n>>> 高精度验证候选点（多组初值测试发现的更低 χ² 点）...")
    chi2_cand_high = verify_chi2(candidate, "候选点", z, mu, cov, bao, eps_high=1e-7)
    chi2_best_high = verify_chi2(best, "主拟合点", z, mu, cov, bao, eps_high=1e-7)

    # 输出候选点的物理量
    A_c, b_c, Om_c, H0_c = candidate
    print("\n候选点 1+w(z) 物理检验:")
    z_check = [0.2, 0.4, 0.6, 0.8, 1.0]
    for zi in z_check:
        wp = one_plus_w(zi, A_c, b_c)
        w = wp - 1.0
        print(f"z={zi:.1f} | 1+w = {wp:.4f}  (w={w:.4f})")

    # ---------- 4. (可选) 多组初值测试和剖面图，默认关闭 ----------
    RUN_MULTI = False   # 耗时，默认关闭
    RUN_PROFILES = False

    if RUN_MULTI:
        N_STARTS = 20
        print(f"\n>>> 多组随机初值测试 ({N_STARTS} 组) ...")
        results = []
        np.random.seed(42)
        for _ in tqdm(range(N_STARTS), desc="Multi-start"):
            init_rnd = [
                np.random.uniform(0.5, 2.5),
                np.random.uniform(20, 32),
                np.random.uniform(0.22, 0.32),
                np.random.uniform(69, 71)
            ]
            res = minimize(lambda p: chi2_total(p, z, mu, cov, bao, False),
                           init_rnd, bounds=bounds_osc, method='L-BFGS-B',
                           options={'maxiter':2000, 'ftol':1e-6})
            if res.success:
                results.append((res.x, res.fun))
        if results:
            chi2_vals = [r[1] for r in results]
            best_local = results[np.argmin(chi2_vals)]
            print(f"最低 χ² = {best_local[1]:.2f}, 参数: A={best_local[0][0]:.4f}, b={best_local[0][1]:.2f}, Om={best_local[0][2]:.4f}, H0={best_local[0][3]:.2f}")
            print(f"χ² 均值 = {np.mean(chi2_vals):.2f}, 标准差 = {np.std(chi2_vals):.2f}")
            close = sum(1 for c in chi2_vals if abs(c - best_local[1]) < 1.0)
            print(f"收敛到最佳 χ² ±1 以内的次数: {close}/{len(results)}")
        else:
            print("警告：所有随机初值优化均失败。")

    if RUN_PROFILES:
        print("\n>>> 生成参数剖面图 (A-H0 等高线 + A 的 Δχ² 曲线) ...")
        # 二维等高线
        A_scan = np.linspace(1.2, 1.6, 25)
        H0_scan = np.linspace(69.5, 70.8, 25)
        chi2_grid = np.zeros((len(A_scan), len(H0_scan)))
        print("  计算二维网格...")
        for i, A in enumerate(tqdm(A_scan, desc="A loop")):
            for j, H0 in enumerate(H0_scan):
                theta = (A, b_best, Om_best, H0)
                chi2_grid[i, j] = chi2_total(theta, z, mu, cov, bao, False)
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(H0_scan, A_scan)
        contour = ax.contour(X, Y, chi2_grid, levels=10, cmap='viridis')
        ax.clabel(contour, inline=True, fontsize=8)
        ax.scatter(H0_best, A_best, color='red', marker='*', s=100, label='best fit')
        ax.set_xlabel('H0 [km/s/Mpc]')
        ax.set_ylabel('A')
        ax.set_title(rf'$\chi^2$ contour (b={b_best:.2f}, $\Omega_m$={Om_best:.4f})')
        ax.legend()
        plt.savefig('chi2_contour.png', dpi=150)
        plt.close()
        print("  保存: chi2_contour.png")

        # 一维 A 扫描
        A_1d = np.linspace(1.2, 1.6, 30)
        chi2_1d = []
        for A in tqdm(A_1d, desc="1D A scan"):
            theta = (A, b_best, Om_best, H0_best)
            chi2_1d.append(chi2_total(theta, z, mu, cov, bao, False))
        chi2_1d = np.array(chi2_1d)
        plt.figure()
        plt.plot(A_1d, chi2_1d - np.min(chi2_1d), 'b-', linewidth=2)
        plt.axhline(1.0, color='r', linestyle='--', label=r'$\Delta\chi^2=1$')
        plt.xlabel('A')
        plt.ylabel(r'$\Delta\chi^2$')
        plt.title(f'1D profile of A (b={b_best:.2f}, $\Omega_m$={Om_best:.4f}, H0={H0_best:.2f})')
        plt.legend()
        plt.grid(True)
        plt.savefig('chi2_profile_A.png', dpi=150)
        plt.close()
        print("  保存: chi2_profile_A.png")

    print("\n所有分析完成。")