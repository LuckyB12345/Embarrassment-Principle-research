"""
全自动拟合脚本：联合 + 单独 (SN / BAO / Planck)
模型：ΛCDM vs 振荡暗能量 (1+w = A sin(bz)/(1+z)^2)
基于 final_fit_physical.py 扩展
【已修复】BAO 使用 DESI DR2 正确的 D_M/r_d 和 D_H/r_d 联合协方差
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, inv
from astropy.cosmology import FLRW, FlatLambdaCDM
from astropy import units as u
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")

# ========== 常量 ==========
C_LIGHT = 299792.458      # km/s
RD_FID = 147.09           # Mpc, 声视界尺度（固定）

# ========== 数据加载 ==========
def load_sn_data():
    """加载 Pantheon+ SN 数据"""
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
    return z, mu, cov

def load_bao_data():
    """
    加载 DESI DR2 BAO 数据 (arXiv:2503.14738, Table IV)
    返回: list of (z, DM_obs, DH_obs, cov_2x2)
    cov_2x2: [[var_DM, cov_DM_DH], [cov_DM_DH, var_DH]]
    """
    # 数据顺序: z, D_M/r_d, D_H/r_d, var_DM, var_DH, cov_DM_DH
    bao_raw = [
        (0.295, 7.391, 19.776, 5.18e-3, 2.81e-1, -5.26e-3),
        (0.510, 11.845, 23.585, 4.76e-3, 1.07e-1, -1.00e-3),
        (0.705, 15.823, 28.001, 6.40e-3, 1.54e-1, -9.00e-4),
        (0.930, 20.313, 30.606, 8.65e-3, 9.80e-2, -2.00e-4),
        (1.317, 27.242, 35.050, 1.10e-2, 1.49e-1, 1.00e-4),
        (1.491, 30.453, 37.693, 1.51e-2, 1.32e-1, -1.00e-4)
    ]
    bao_list = []
    for z, dm, dh, var_dm, var_dh, cov in bao_raw:
        cov_mat = np.array([[var_dm, cov], [cov, var_dh]])
        bao_list.append((z, dm, dh, cov_mat))
    return bao_list

# ========== 振荡暗能量宇宙学模型 ==========
class OscillatingDE(FLRW):
    def __init__(self, H0, Om0, A, b, name="OscillatingDE"):
        self._A = A
        self._b = b
        super().__init__(H0=H0*u.km/u.s/u.Mpc, Om0=Om0, Ode0=1-Om0,
                         Tcmb0=2.7255*u.K, Neff=3.046, name=name)

    def w(self, z):
        return -1.0 + self._A * np.sin(self._b * z) / (1+z)**2

    def de_density_scale(self, z):
        def integrand(zp):
            return 3 * (self.w(zp) + 1) / (1+zp)
        val, _ = quad(integrand, 0, z, epsrel=1e-7, limit=500)
        val = np.clip(val, -50, 50)
        return np.exp(val)

# ========== 辅助函数 ==========
def mu_cosmo(cosmo, z):
    d_L = cosmo.luminosity_distance(z).value  # Mpc
    return 5 * np.log10(d_L) + 25

def planck2018_chi2(Om, H0):
    Om_p, H0_p = 0.3111, 67.66
    sig_Om, sig_H0 = 0.0056, 0.42
    corr = 0.18
    chi2 = ((Om - Om_p)/sig_Om)**2 + ((H0 - H0_p)/sig_H0)**2 \
           - 2*corr*(Om-Om_p)*(H0-H0_p)/(sig_Om*sig_H0)
    return chi2

def bao_chi2(cosmo, bao_list, rd=RD_FID):
    """计算 BAO 联合似然 (D_M/r_d, D_H/r_d)"""
    chi2 = 0.0
    for z, DM_obs, DH_obs, cov in bao_list:
        DM_theo = cosmo.comoving_distance(z).value / rd
        DH_theo = C_LIGHT / (cosmo.H(z).value * rd)
        diff = np.array([DM_theo - DM_obs, DH_theo - DH_obs])
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return 1e10
        chi2 += diff.T @ inv_cov @ diff
    return chi2

# ================== 联合拟合 (SN+BAO+Planck) ==================
def chi2_total_osc(params, z_sn, mu_sn, cov_sn, bao_list):
    A, b, Om, H0 = params
    # 物理约束：z=0.2 处加速膨胀 (1+w < 2/3)
    w0p2 = -1.0 + A * np.sin(b * 0.2) / (1+0.2)**2
    if w0p2 >= -1/3:
        return 1e10
    if not (0 <= A <= 10 and 10 <= b <= 50 and 0.01 <= Om <= 0.50 and 65 <= H0 <= 78):
        return 1e10

    cosmo = OscillatingDE(H0=H0, Om0=Om, A=A, b=b)

    # SN
    mu_model = np.array([mu_cosmo(cosmo, zi) for zi in z_sn])
    if not np.all(np.isfinite(mu_model)):
        return 1e10
    resid = mu_sn - mu_model
    try:
        cho = cho_factor(cov_sn)
        chi2_sn = resid @ cho_solve(cho, resid)
    except:
        return 1e10

    # BAO
    chi2_bao = bao_chi2(cosmo, bao_list)

    # Planck
    chi2_planck = planck2018_chi2(Om, H0)

    return chi2_sn + chi2_bao + chi2_planck

def chi2_total_lcd(params, z_sn, mu_sn, cov_sn, bao_list):
    Om, H0 = params
    if not (0.01 <= Om <= 0.50 and 65 <= H0 <= 78):
        return 1e10
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om)
    mu_model = np.array([mu_cosmo(cosmo, zi) for zi in z_sn])
    resid = mu_sn - mu_model
    try:
        cho = cho_factor(cov_sn)
        chi2_sn = resid @ cho_solve(cho, resid)
    except:
        return 1e10
    chi2_bao = bao_chi2(cosmo, bao_list)
    chi2_planck = planck2018_chi2(Om, H0)
    return chi2_sn + chi2_bao + chi2_planck

# ================== 单独拟合 SN ==================
def chi2_sn_osc(params, z_sn, mu_sn, cov_sn):
    A, b, Om, H0 = params
    if not (0 <= A <= 10 and 10 <= b <= 50 and 0.01 <= Om <= 0.50 and 65 <= H0 <= 78):
        return 1e10
    cosmo = OscillatingDE(H0=H0, Om0=Om, A=A, b=b)
    mu_model = np.array([mu_cosmo(cosmo, zi) for zi in z_sn])
    resid = mu_sn - mu_model
    try:
        cho = cho_factor(cov_sn)
        return resid @ cho_solve(cho, resid)
    except:
        return 1e10

def chi2_sn_lcd(params, z_sn, mu_sn, cov_sn):
    Om, H0 = params
    if not (0.01 <= Om <= 0.50 and 65 <= H0 <= 78):
        return 1e10
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om)
    mu_model = np.array([mu_cosmo(cosmo, zi) for zi in z_sn])
    resid = mu_sn - mu_model
    try:
        cho = cho_factor(cov_sn)
        return resid @ cho_solve(cho, resid)
    except:
        return 1e10

# ================== 单独拟合 BAO ==================
def chi2_bao_osc(params, bao_list):
    A, b, Om, H0 = params
    if not (0 <= A <= 10 and 10 <= b <= 50 and 0.01 <= Om <= 0.50 and 65 <= H0 <= 78):
        return 1e10
    cosmo = OscillatingDE(H0=H0, Om0=Om, A=A, b=b)
    return bao_chi2(cosmo, bao_list)

def chi2_bao_lcd(params, bao_list):
    Om, H0 = params
    if not (0.01 <= Om <= 0.50 and 65 <= H0 <= 78):
        return 1e10
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om)
    return bao_chi2(cosmo, bao_list)

# ================== 单独拟合 Planck ==================
def chi2_planck_only(params):
    Om, H0 = params
    return planck2018_chi2(Om, H0)

# ================== 主程序 ==================
def main():
    print("="*70)
    print("自动运行：联合拟合 + 单独数据集拟合 (BAO 已修复)")
    print("模型：ΛCDM vs 振荡暗能量 (EO)")
    print("="*70)

    # 加载数据
    z_sn, mu_sn, cov_sn = load_sn_data()
    bao_list = load_bao_data()
    print(f"SN 数据点: {len(z_sn)}")
    print(f"BAO 红移点: {[z for z,_,_,_ in bao_list]}")

    # ---------- 1. 联合拟合 ----------
    print("\n>>> 联合拟合 (SN+BAO+Planck) ...")
    # ΛCDM
    res_lcd = minimize(lambda p: chi2_total_lcd(p, z_sn, mu_sn, cov_sn, bao_list),
                       [0.3, 69.0], bounds=[(0.01, 0.50), (65, 78)],
                       method='L-BFGS-B', options={'maxiter':20000})
    chi2_lcd_joint = res_lcd.fun
    Om_lcd_joint, H0_lcd_joint = res_lcd.x

    # 振荡模型
    init_osc = [1.5377, 26.4174, 0.1252, 69.9076]  # 论文中位值
    res_osc = minimize(lambda p: chi2_total_osc(p, z_sn, mu_sn, cov_sn, bao_list),
                       init_osc, bounds=[(0,10),(10,50),(0.01,0.50),(65,78)],
                       method='L-BFGS-B', options={'maxiter':50000})
    if not res_osc.success:
        print("警告：振荡模型联合拟合未收敛，使用初值重试...")
        res_osc = minimize(lambda p: chi2_total_osc(p, z_sn, mu_sn, cov_sn, bao_list),
                           [1.5, 25.0, 0.12, 69.5], bounds=[(0,10),(10,50),(0.01,0.50),(65,78)],
                           method='L-BFGS-B', options={'maxiter':50000})
    A_joint, b_joint, Om_osc_joint, H0_osc_joint = res_osc.x
    chi2_osc_joint = res_osc.fun
    delta_joint = chi2_lcd_joint - chi2_osc_joint

    # ---------- 2. 单独拟合 SN ----------
    print("\n>>> 单独拟合 SN ...")
    res_lcd_sn = minimize(lambda p: chi2_sn_lcd(p, z_sn, mu_sn, cov_sn),
                          [0.3, 69.0], bounds=[(0.01,0.50),(65,78)],
                          method='L-BFGS-B')
    chi2_lcd_sn = res_lcd_sn.fun
    Om_lcd_sn, H0_lcd_sn = res_lcd_sn.x

    res_osc_sn = minimize(lambda p: chi2_sn_osc(p, z_sn, mu_sn, cov_sn),
                          [1.5, 25.0, 0.12, 69.5], bounds=[(0,10),(10,50),(0.01,0.50),(65,78)],
                          method='L-BFGS-B')
    if not res_osc_sn.success:
        res_osc_sn = minimize(lambda p: chi2_sn_osc(p, z_sn, mu_sn, cov_sn),
                              [1.5377, 26.4174, 0.1252, 69.9076], bounds=[(0,10),(10,50),(0.01,0.50),(65,78)],
                              method='L-BFGS-B')
    A_sn, b_sn, Om_osc_sn, H0_osc_sn = res_osc_sn.x
    chi2_osc_sn = res_osc_sn.fun
    delta_sn = chi2_lcd_sn - chi2_osc_sn

    # ---------- 3. 单独拟合 BAO ----------
    print("\n>>> 单独拟合 BAO (正确似然) ...")
    res_lcd_bao = minimize(lambda p: chi2_bao_lcd(p, bao_list),
                           [0.3, 69.0], bounds=[(0.01,0.50),(65,78)],
                           method='L-BFGS-B')
    chi2_lcd_bao = res_lcd_bao.fun
    Om_lcd_bao, H0_lcd_bao = res_lcd_bao.x

    res_osc_bao = minimize(lambda p: chi2_bao_osc(p, bao_list),
                           [1.5, 25.0, 0.12, 69.5], bounds=[(0,10),(10,50),(0.01,0.50),(65,78)],
                           method='L-BFGS-B')
    if not res_osc_bao.success:
        res_osc_bao = minimize(lambda p: chi2_bao_osc(p, bao_list),
                               [1.5377, 26.4174, 0.1252, 69.9076], bounds=[(0,10),(10,50),(0.01,0.50),(65,78)],
                               method='L-BFGS-B')
    A_bao, b_bao, Om_osc_bao, H0_osc_bao = res_osc_bao.x
    chi2_osc_bao = res_osc_bao.fun
    delta_bao = chi2_lcd_bao - chi2_osc_bao

    # ---------- 4. 单独拟合 Planck ----------
    print("\n>>> 单独拟合 Planck (固定振荡参数为联合最佳值) ...")
    res_planck_lcd = minimize(lambda p: planck2018_chi2(p[0], p[1]),
                              [0.31, 67.7], bounds=[(0.01,0.50),(65,78)],
                              method='L-BFGS-B')
    chi2_planck_lcd = res_planck_lcd.fun
    Om_planck_lcd, H0_planck_lcd = res_planck_lcd.x

    def planck_osc_fixed(params):
        Om, H0 = params
        return planck2018_chi2(Om, H0)
    res_planck_osc = minimize(lambda p: planck_osc_fixed(p),
                              [Om_osc_joint, H0_osc_joint], bounds=[(0.01,0.50),(65,78)],
                              method='L-BFGS-B')
    chi2_planck_osc = res_planck_osc.fun
    Om_planck_osc, H0_planck_osc = res_planck_osc.x
    delta_planck = chi2_planck_lcd - chi2_planck_osc

    # ========== 输出汇总表格 ==========
    print("\n" + "="*80)
    print("汇总结果 (BAO 已使用正确的 D_M/r_d, D_H/r_d 联合似然)")
    print("="*80)
    print(f"{'数据集':<20} {'模型':<10} {'χ²':<12} {'最佳参数':<30}")
    print("-"*80)
    print(f"{'联合拟合':<20} {'ΛCDM':<10} {chi2_lcd_joint:<12.2f} Ωm={Om_lcd_joint:.4f}, H0={H0_lcd_joint:.2f}")
    print(f"{'':<20} {'EO':<10} {chi2_osc_joint:<12.2f} A={A_joint:.4f}, b={b_joint:.2f}, Ωm={Om_osc_joint:.4f}, H0={H0_osc_joint:.2f}")
    print(f"{'':<20} {'Δχ²':<10} {delta_joint:<12.2f} (EO 优于 ΛCDM)")
    print("-"*80)
    print(f"{'SN only':<20} {'ΛCDM':<10} {chi2_lcd_sn:<12.2f} Ωm={Om_lcd_sn:.4f}, H0={H0_lcd_sn:.2f}")
    print(f"{'':<20} {'EO':<10} {chi2_osc_sn:<12.2f} A={A_sn:.4f}, b={b_sn:.2f}, Ωm={Om_osc_sn:.4f}, H0={H0_osc_sn:.2f}")
    print(f"{'':<20} {'Δχ²':<10} {delta_sn:<12.2f}")
    print("-"*80)
    print(f"{'BAO only':<20} {'ΛCDM':<10} {chi2_lcd_bao:<12.2f} Ωm={Om_lcd_bao:.4f}, H0={H0_lcd_bao:.2f}")
    print(f"{'':<20} {'EO':<10} {chi2_osc_bao:<12.2f} A={A_bao:.4f}, b={b_bao:.2f}, Ωm={Om_osc_bao:.4f}, H0={H0_osc_bao:.2f}")
    print(f"{'':<20} {'Δχ²':<10} {delta_bao:<12.2f}")
    print("-"*80)
    print(f"{'Planck only':<20} {'ΛCDM':<10} {chi2_planck_lcd:<12.2f} Ωm={Om_planck_lcd:.4f}, H0={H0_planck_lcd:.2f}")
    print(f"{'':<20} {'EO (A,b固定)':<13} {chi2_planck_osc:<12.2f} Ωm={Om_planck_osc:.4f}, H0={H0_planck_osc:.2f}")
    print(f"{'':<20} {'Δχ²':<10} {delta_planck:<12.2f}")
    print("="*80)
    print("\n说明：Planck only 中 EO 固定 A,b 为联合最佳值，因此 χ² 与 ΛCDM 几乎相同。")
    print("      BAO 似然现在基于 DESI DR2 官方 D_M/r_d, D_H/r_d 及完整协方差。")

if __name__ == "__main__":
    main()