import numpy as np
np.random.seed(42)
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import quad
from scipy.interpolate import interp1d
from functools import lru_cache
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

C_LIGHT = 299792.458      # km/s
RD_FID = 147.78            # Mpc (Planck 2018 fiducial)

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

    print("协方差对角线前5项:", np.diag(cov)[:5])
    print("mu 前5项:", mu[:5])

    bao = {
        0.38: (10.23, 0.08),
        0.51: (13.41, 0.09),
        0.61: (15.94, 0.10)
    }
    return z, mu, cov, bao

def E_lcdm(z, Om):
    return np.sqrt(Om*(1+z)**3 + (1-Om))

def mu_lcdm(z, Om, H0):
    try:
        dc, _ = quad(lambda x: C_LIGHT / E_lcdm(x, Om), 0, z, limit=200, epsrel=1e-6)
    except:
        return 1e10
    dc /= H0
    return 5*np.log10(dc * (1+z)) + 25

def DM_H_z_lcdm(z, Om, H0):
    try:
        dc, _ = quad(lambda x: C_LIGHT / E_lcdm(x, Om), 0, z, limit=200, epsrel=1e-6)
    except:
        return 1e10, 1e10
    dc /= H0
    return dc, H0 * E_lcdm(z, Om)

@lru_cache(maxsize=256)
def rho_osc(A, a, b):
    """
    暗能量密度比 ρ_de(z)/ρ_de(0) = exp( +3A ∫ e^{-ax} sin(bx)/(1+x) dx )
    符号采用正号，符合数学推导和物理图像 (w > -1 时过去密度更高)
    """
    zs = np.linspace(0, 2.5, 250)
    rho = np.zeros_like(zs)
    for i, zz in enumerate(zs):
        try:
            val, _ = quad(lambda x: 3*A*np.exp(-a*x)*np.sin(b*x)/(1+x), 0, zz,
                          limit=200, epsrel=1e-6)
        except:
            val = -np.inf
        if val < -100:
            val = -100
        rho[i] = np.exp(val)
    return interp1d(zs, rho, kind='cubic', fill_value='extrapolate')

def mu_osc(z, A, a, b, Om, H0):
    rho = rho_osc(A, a, b)
    def integrand(x):
        rho_x = rho(x)
        if rho_x < 0:
            rho_x = 0.0          # 物理上不允许负能量密度，截断
        return C_LIGHT / np.sqrt(Om*(1+x)**3 + (1-Om)*rho_x)
    try:
        dc, _ = quad(integrand, 0, z, limit=200, epsrel=1e-6)
    except:
        return 1e10
    return 5*np.log10((dc / H0) * (1+z)) + 25

def DM_H_z_osc(z, A, a, b, Om, H0):
    rho = rho_osc(A, a, b)
    def integrand(x):
        rho_x = rho(x)
        if rho_x < 0:
            rho_x = 0.0
        return C_LIGHT / np.sqrt(Om*(1+x)**3 + (1-Om)*rho_x)
    try:
        dc, _ = quad(integrand, 0, z, limit=200, epsrel=1e-6)
    except:
        return 1e10, 1e10
    DM = dc / H0
    rho_z = rho(z)
    if rho_z < 0:
        rho_z = 0.0
    E_z = np.sqrt(Om*(1+z)**3 + (1-Om)*rho_z)
    return DM, H0 * E_z

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
    for z_bao, (obs, err) in bao.items():
        if is_lcdm:
            D_M, H_z = DM_H_z_lcdm(z_bao, Om, H0)
        else:
            A, a, b, Om, H0 = theta
            D_M, H_z = DM_H_z_osc(z_bao, A, a, b, Om, H0)
        if D_M > 1e9 or H_z < 1e-10 or not np.isfinite(D_M) or not np.isfinite(H_z):
            return 1e10
        D_V = (D_M**2 * C_LIGHT * z_bao / H_z) ** (1/3)
        if not np.isfinite(D_V) or D_V > 10000:
            return 1e10
        model_val = D_V / RD_FID
        chi2_bao += ((model_val - obs) / err) ** 2

    return chi2_sn + chi2_bao

def fit_one(z, mu, cov, bao, Om_bounds, fix_Om=False, Om_fixed=None):
    rho_osc.cache_clear()
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

if __name__ == "__main__":
    z, mu, cov, bao = load_data()

    constraints = [
        {"name": "无约束 (Ωm ≥ 0.1)",   "Om_bounds": (0.1, 0.5), "fix_Om": False},
        {"name": "Ωm ≥ 0.2",           "Om_bounds": (0.2, 0.5), "fix_Om": False},
        {"name": "Ωm ≥ 0.25",          "Om_bounds": (0.25, 0.5), "fix_Om": False},
        {"name": "Ωm 固定 = 0.315",     "Om_bounds": None, "fix_Om": True, "Om_fixed": 0.315}
    ]

    results = []
    print("\n" + "="*80)
    print("最终物理正确版 (积分使用 +3A，协方差不缩放)")
    print("="*80)

    start_time = time.time()
    for constr in constraints:
        print(f"\n>>> {constr['name']}")
        if constr['fix_Om']:
            Om_l, H0_l, chi2_l, A, a, b, Om_o, H0_o, chi2_o = fit_one(
                z, mu, cov, bao, None, fix_Om=True, Om_fixed=constr['Om_fixed'])
        else:
            Om_l, H0_l, chi2_l, A, a, b, Om_o, H0_o, chi2_o = fit_one(
                z, mu, cov, bao, constr['Om_bounds'], fix_Om=False)

        delta = chi2_l - chi2_o
        print(f"ΛCDM     : Ωm={Om_l:.4f}, H0={H0_l:.2f}, χ²={chi2_l:.2f}")
        print(f"振荡模型 : A={A:.4f}, α={a:.4f}, β={b:.2f}, Ωm={Om_o:.4f}, H0={H0_o:.2f}, χ²={chi2_o:.2f}")
        print(f"Δχ² = {delta:.2f}")

        N = len(z) + len(bao)
        k_lcdm, k_osc = 2, 5
        aic_l = chi2_l + 2*k_lcdm
        aic_o = chi2_o + 2*k_osc
        bic_l = chi2_l + k_lcdm*np.log(N)
        bic_o = chi2_o + k_osc*np.log(N)
        print(f"AIC: ΛCDM={aic_l:.2f}, 振荡={aic_o:.2f} (ΔAIC={aic_l - aic_o:.2f})")
        print(f"BIC: ΛCDM={bic_l:.2f}, 振荡={bic_o:.2f} (ΔBIC={bic_l - bic_o:.2f})")

        results.append({
            "constraint": constr['name'],
            "Om_l": Om_l, "H0_l": H0_l, "chi2_l": chi2_l,
            "A": A, "alpha": a, "beta": b,
            "Om_o": Om_o, "H0_o": H0_o, "chi2_o": chi2_o,
            "delta": delta, "aic_l": aic_l, "aic_o": aic_o, "bic_l": bic_l, "bic_o": bic_o
        })

    elapsed = time.time() - start_time
    print(f"\n⏱️ 总耗时: {elapsed:.1f} 秒")
    print("="*80)

    # 汇总表格
    print("\n" + "="*80)
    print("汇总表格")
    print("="*80)
    print("{:<20} {:<6} {:<6} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
        "约束", "Ωm_L", "H0_L", "χ²_L", "A", "Ωm_O", "H0_O", "χ²_O", "Δχ²"))
    for r in results:
        print("{:<20} {:<6.4f} {:<6.2f} {:<8.2f} {:<8.4f} {:<8.4f} {:<8.2f} {:<8.2f} {:<8.2f}".format(
            r['constraint'], r['Om_l'], r['H0_l'], r['chi2_l'],
            r['A'], r['Om_o'], r['H0_o'], r['chi2_o'], r['delta']))

    # ======================= 绘图（使用无约束结果） =======================
    if results:
        best = results[0]
        A_bf, a_bf, b_bf = best['A'], best['alpha'], best['beta']
        Om_bf, H0_bf = best['Om_o'], best['H0_o']
        Om_lcdm, H0_lcdm = best['Om_l'], best['H0_l']

        def rho_plot_high(A, a, b):
            zs = np.linspace(0, 2.5, 600)
            rho = np.zeros_like(zs)
            for i, zz in enumerate(zs):
                try:
                    val, _ = quad(lambda x: 3*A*np.exp(-a*x)*np.sin(b*x)/(1+x), 0, zz,
                                  limit=500, epsrel=1e-9)
                except:
                    val = -np.inf
                if val < -100:
                    val = -100
                rho[i] = np.exp(val)
            return interp1d(zs, rho, kind='cubic', fill_value='extrapolate')

        rho_high = rho_plot_high(A_bf, a_bf, b_bf)
        z_arr = np.linspace(0, 1.2, 200)

        # 残差图
        mu_osc_arr = np.array([mu_osc(zi, A_bf, a_bf, b_bf, Om_bf, H0_bf) for zi in z])
        mu_lcdm_arr = np.array([mu_lcdm(zi, Om_lcdm, H0_lcdm) for zi in z])
        plt.figure(figsize=(10,5))
        plt.scatter(z, mu - mu_osc_arr, s=2, alpha=0.5, label='Oscillating')
        plt.scatter(z, mu - mu_lcdm_arr, s=2, alpha=0.5, label='ΛCDM')
        plt.axhline(0, c='k')
        plt.xlabel('Redshift z')
        plt.ylabel(r'$\mu_{\rm obs} - \mu_{\rm model}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig('fig_residual.png', dpi=200)
        plt.close()

        # w(z) 图
        wz = [-1 + A_bf * np.exp(-a_bf*z) * np.sin(b_bf*z) for z in z_arr]
        plt.figure(figsize=(8,5))
        plt.plot(z_arr, wz, lw=2)
        plt.axhline(-1, c='r', ls='--')
        plt.xlabel('z')
        plt.ylabel('w(z)')
        plt.tight_layout()
        plt.savefig('fig_wz.png', dpi=200)
        plt.close()

        # H(z) 图
        H_osc = []
        for zz in z_arr:
            rho_val = rho_high(zz)
            if rho_val < 0:
                rho_val = 0.0
            H_osc.append(H0_bf * np.sqrt(Om_bf*(1+zz)**3 + (1-Om_bf)*rho_val))
        H_lcd = [H0_lcdm * np.sqrt(Om_lcdm*(1+zz)**3 + (1-Om_lcdm)) for zz in z_arr]
        plt.figure(figsize=(8,5))
        plt.plot(z_arr, H_osc, lw=2, label='Oscillating')
        plt.plot(z_arr, H_lcd, '--', lw=2, label='ΛCDM')
        plt.xlabel('z')
        plt.ylabel(r'$H(z)$ [km/s/Mpc]')
        plt.legend()
        plt.tight_layout()
        plt.savefig('fig_Hz.png', dpi=200)
        plt.close()

        print("\n✅ 运行完成。效果图已保存: fig_residual.png, fig_wz.png, fig_Hz.png")
    else:
        print("❌ 拟合失败，请检查数据文件路径或参数边界。")