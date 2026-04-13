import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")

C_LIGHT = 299792.458  # km/s

# ================= 1. 加载 Pantheon+ 数据 =================
def load_pantheon_full():
    data = np.genfromtxt("pantheon+_data.txt", skip_header=1, usecols=(4,10,11))
    z, mu_obs, mu_err = data.T
    msk = (z > 0) & np.isfinite(mu_obs)
    z = z[msk]
    mu_obs = mu_obs[msk]
    mu_err = mu_err[msk]

    with open("Pantheon+SH0ES_STAT+SYS.cov", 'r') as f:
        n = int(f.readline().strip())
    cov_flat = np.loadtxt("Pantheon+SH0ES_STAT+SYS.cov", skiprows=1)
    cov_full = cov_flat.reshape(n, n)
    cov_full = (cov_full + cov_full.T) / 2
    cov_sub = cov_full[np.ix_(msk, msk)]

    diag_cov = np.diag(cov_sub)
    ratio = np.median(diag_cov / (mu_err**2))
    if ratio > 1e4:
        cov_sub /= 1e6
        print(f"协方差单位修正 (mmag^2 -> mag^2), 比值={ratio:.1e}")
    else:
        print(f"协方差对角元/err^2 中位数: {ratio:.2f}，单位正确")
    print(f"有效 SNe 数量: {len(z)}")
    return z, mu_obs, cov_sub

z_sn, mu_o, cov_sn = load_pantheon_full()

try:
    cho_fac = cho_factor(cov_sn)
except LinAlgError:
    cov_sn += np.eye(len(cov_sn)) * 1e-10
    cho_fac = cho_factor(cov_sn)

# ================= 2. 宇宙学模型 =================
def E_lcdm(z, Om):
    return np.sqrt(Om * (1+z)**3 + (1 - Om))

def mu_lcdm(z, Om, H0):
    def integrand(x):
        return C_LIGHT / E_lcdm(x, Om)
    dc, _ = quad(integrand, 0, z, limit=200)
    dc /= H0
    dL = dc * (1+z)
    return 5 * np.log10(dL) + 25

def mu_osc(z, A, a, b, Om, H0):
    def integrand_rho(zp):
        return 3.0 * A * np.exp(-a * zp) * np.sin(b * zp) / (1 + zp)
    ln_rho, _ = quad(integrand_rho, 0, z, limit=200)
    rho_de = np.exp(ln_rho)
    def E_z(zp):
        return np.sqrt(Om * (1+zp)**3 + (1 - Om) * rho_de)
    dc, _ = quad(lambda x: C_LIGHT / E_z(x), 0, z, limit=200)
    dc /= H0
    dL = dc * (1+z)
    return 5 * np.log10(dL) + 25

# ================= 3. BAO =================
bao_data = {0.38: (1512.39, 12), 0.51: (1975.22, 13), 0.61: (2347.61, 14)}
rs_fid = 147.78

def chi2_bao(theta):
    A, a, b, Om, H0 = theta
    chi2 = 0.0
    for z, (obs, err) in bao_data.items():
        if A == 0:
            mu = mu_lcdm(z, Om, H0)
            E = E_lcdm(z, Om)
        else:
            mu = mu_osc(z, A, a, b, Om, H0)
            ln_rho, _ = quad(lambda zp: 3*A*np.exp(-a*zp)*np.sin(b*zp)/(1+zp), 0, z)
            rho_de = np.exp(ln_rho)
            E = np.sqrt(Om*(1+z)**3 + (1-Om)*rho_de)
        dL = 10 ** ((mu - 25) / 5)
        D_M = dL / (1 + z)
        H_z = H0 * E
        D_V = (D_M**2 * C_LIGHT * z / H_z) ** (1/3)
        chi2 += ((D_V / rs_fid - obs) / err) ** 2
    return chi2

def chi2_sn(theta):
    A, a, b, Om, H0 = theta
    if A == 0:
        mu_model = np.array([mu_lcdm(zi, Om, H0) for zi in z_sn])
    else:
        mu_model = np.array([mu_osc(zi, A, a, b, Om, H0) for zi in z_sn])
    resid = mu_o - mu_model
    return resid.T @ cho_solve(cho_fac, resid)

def chi2_joint(theta):
    return chi2_sn(theta) + chi2_bao(theta)

# ================= 4. 拟合 =================
def fit_osc(x0, bounds):
    res = minimize(chi2_joint, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 800, 'ftol': 1e-9})
    return res.x, res.fun

def fit_lcdm():
    best_chi2 = np.inf
    best_params = None
    inits = [(0.30, 70.0), (0.32, 68.0), (0.34, 72.0)]
    for Om_init, H0_init in inits:
        def chi2(params):
            Om, H0 = params
            return chi2_joint([0, 0, 0, Om, H0])
        res = minimize(chi2, [Om_init, H0_init], bounds=[(0.2,0.45),(60,80)], method='L-BFGS-B')
        if res.fun < best_chi2:
            best_chi2 = res.fun
            best_params = res.x
    return best_params[0], best_params[1], best_chi2

# ================= 5. 运行拟合 =================
print("="*70)
print("1. 自由拟合")
x0_free = [0.5, 0.5, 20.0, 0.31, 68.0]
bounds_free = [(0, 0.6), (0, 5), (0, 25), (0.2, 0.45), (60, 80)]
best_osc, chi2_osc = fit_osc(x0_free, bounds_free)
A_bf, a_bf, b_bf, Om_bf, H0_bf = best_osc

print("2. ΛCDM 拟合")
Om_lcdm, H0_lcdm, chi2_lcdm = fit_lcdm()

delta_chi2 = chi2_lcdm - chi2_osc
k_extra = 3
n_sn = len(z_sn)
bic = delta_chi2 - k_extra * np.log(n_sn)
aic = delta_chi2 - 2 * k_extra

print("3. 保守拟合")
x0_con = [0.5, 0.5, 20.0, 0.31, 68.0]
bounds_con = [(0, 0.6), (0.1, 5), (0, 25), (0.2, 0.45), (60, 80)]
best_con, chi2_con = fit_osc(x0_con, bounds_con)

# ================= 6. 自动出图（论文级别 4 张完整版）=================
def make_all_plots():
    plt.rcParams.update({'font.size': 12})

    # 图1：残差比较
    mu_osc_vals = np.array([mu_osc(z, A_bf,a_bf,b_bf,Om_bf,H0_bf) for z in z_sn])
    mu_lcdm_vals = np.array([mu_lcdm(z, Om_lcdm, H0_lcdm) for z in z_sn])
    res_osc = mu_o - mu_osc_vals
    res_lcdm = mu_o - mu_lcdm_vals

    plt.figure(figsize=(10,5))
    plt.scatter(z_sn, res_osc, s=2, alpha=0.5, label="Oscillating model")
    plt.scatter(z_sn, res_lcdm, s=2, alpha=0.5, label="ΛCDM")
    plt.axhline(0, c='k', lw=1)
    plt.xlabel("Redshift z")
    plt.ylabel(r"$\mu_{\rm obs} - \mu_{\rm model}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig1_residual.png", dpi=200)
    plt.close()

    # 图2：w(z) 暗能量状态方程
    z_arr = np.linspace(0, 1.2, 100)
    w_osc = []
    for z in z_arr:
        w = -1 + A_bf * np.exp(-a_bf*z) * np.sin(b_bf*z)
        w_osc.append(w)
    plt.figure(figsize=(8,5))
    plt.plot(z_arr, w_osc, label="Oscillating w(z)", lw=2)
    plt.axhline(-1, c='r', ls='--', label="ΛCDM w=-1")
    plt.xlabel("z")
    plt.ylabel("w(z)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig2_wz.png", dpi=200)
    plt.close()

    # 图3：H(z) 哈勃参数演化
    H_osc = []
    H_lcd = []
    for z in z_arr:
        # 振荡模型 H(z)
        ln_rho, _ = quad(lambda zp: 3*A_bf*np.exp(-a_bf*zp)*np.sin(b_bf*zp)/(1+zp), 0, z)
        rho_de = np.exp(ln_rho)
        E_osc = np.sqrt(Om_bf*(1+z)**3 + (1-Om_bf)*rho_de)
        H_osc.append(H0_bf * E_osc)

        # ΛCDM H(z)
        E_lcd = np.sqrt(Om_lcdm*(1+z)**3 + (1-Om_lcdm))
        H_lcd.append(H0_lcdm * E_lcd)

    plt.figure(figsize=(8,5))
    plt.plot(z_arr, H_osc, label="Oscillating model", lw=2)
    plt.plot(z_arr, H_lcd, '--', label="ΛCDM", lw=2)
    plt.xlabel("z")
    plt.ylabel("H(z) [km/s/Mpc]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig3_Hz.png", dpi=200)
    plt.close()

    # 图4：χ² 对比汇总图
    plt.figure(figsize=(7,4))
    plt.bar(["LCDM", "Osc"], [chi2_lcdm, chi2_osc], color=['#ff6666','#6699ff'])
    plt.ylabel(r"$\chi^2_{\rm joint}$")
    plt.title(r"$\Delta\chi^2 = %.2f$" % delta_chi2)
    plt.tight_layout()
    plt.savefig("fig4_chi2_summary.png", dpi=200)
    plt.close()

    print("✅ 4 张论文图已全部生成：fig1~fig4")

make_all_plots()

# ================= 最终输出 =================
print("\n" + "="*70)
print(f"Δχ² = {delta_chi2:.2f}")
print(f"ΔBIC = {bic:.2f}")
print(f"ΔAIC = {aic:.2f}")
print("="*70)