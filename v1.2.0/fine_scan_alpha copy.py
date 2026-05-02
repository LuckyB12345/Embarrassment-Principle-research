"""
检查 α = -12 时暗能量密度的高红移行为
使用最佳拟合参数：A=0.0045, α=-12.0, β=17.85, Ωm=0.0484, H0=75.94
计算 ρ_de(z) / ρ_de(0)，并与物质密度 (Ωm*(1+z)^3) 和大致辐射密度对比。
"""

import numpy as np
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")

# ---------- 常数 ----------
C_LIGHT = 299792.458          # 光速 (km/s)
H0_BEST = 75.94               # km/s/Mpc
Omega_m_best = 0.0484
Omega_de0 = 1.0 - Omega_m_best  # 平坦宇宙，Ωk=0
A_best = 0.0045
alpha_best = -12.0
beta_best = 17.85

# 积分精度设置（与主拟合一致）
INTEGRAL_EPSREL = 1e-7
INTEGRAL_LIMIT = 500

# 用于临界密度换算的 H0 比例（无关紧要，这里只比较密度相对值）
# 今天临界密度 ρ_crit0 = 3H0^2/(8πG)，我们只需要相对于今天暗能量密度的比值

def rho_de_over_rho_de0(z, A, a, b):
    """返回 ρ_de(z) / ρ_de(0)"""
    if z == 0.0:
        return 1.0
    try:
        val, _ = quad(
            lambda x: 3*A*np.exp(-a*x)*np.sin(b*x)/(1+x),
            0, z,
            limit=INTEGRAL_LIMIT,
            epsrel=INTEGRAL_EPSREL
        )
    except:
        return np.inf
    # 处理极端负值
    if val < -300:
        val = -300
    return np.exp(val)

def matter_density_factor(z, Om0):
    """物质密度 ρ_m(z) / ρ_crit0 = Om0 * (1+z)^3"""
    return Om0 * (1+z)**3

def radiation_density_factor(z, Om0, H0):
    """
    辐射密度 ρ_r(z) / ρ_crit0 ≈ Ω_r0 * (1+z)^4
    Ω_r0 取 Planck 典型值 ~ 9e-5 （含中微子效应）
    """
    Omega_r0 = 9.0e-5
    return Omega_r0 * (1+z)**4

def main():
    z_list = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.34, 3.0, 5.0, 10.0])
    # Pantheon+ 最大红移约 2.34
    # CMB 红移 ≈ 1090

    print("=" * 80)
    print("暗能量密度 (相对于今天临界密度) 在高红移的行为")
    print(f"模型参数: A={A_best}, α={alpha_best}, β={beta_best}")
    print(f"Ωm={Omega_m_best}, H0={H0_BEST} km/s/Mpc")
    print("=" * 80)
    print(f"{'z':<8} {'ρ_de(z)/ρ_crit0':<18} {'ρ_m(z)/ρ_crit0':<18} {'ρ_de/ρ_m':<12}")
    print("-" * 56)

    for z in z_list:
        rde_ratio = rho_de_over_rho_de0(z, A_best, alpha_best, beta_best)
        # ρ_de(z) / ρ_crit0 = Omega_de0 * rde_ratio (因为 ρ_de0 = Ω_de0 * ρ_crit0)
        rde_full = Omega_de0 * rde_ratio
        rm_full = matter_density_factor(z, Omega_m_best)
        ratio = rde_full / rm_full if rm_full > 0 else np.inf
        print(f"{z:<8.2f} {rde_full:<18.6e} {rm_full:<18.6e} {ratio:<12.4e}")

    print("\n" + "=" * 80)
    print("检查在 CMB 红移 (z~1090) 附近的行为")
    z_cmb = 1090.0
    rde_cmb_ratio = rho_de_over_rho_de0(z_cmb, A_best, alpha_best, beta_best)
    rde_cmb = Omega_de0 * rde_cmb_ratio
    rm_cmb = matter_density_factor(z_cmb, Omega_m_best)
    rad_cmb = radiation_density_factor(z_cmb, Omega_m_best, H0_BEST)
    print(f"z = {z_cmb}")
    print(f"ρ_de/ρ_crit0 = {rde_cmb:.4e}")
    print(f"ρ_m /ρ_crit0 = {rm_cmb:.4e}")
    print(f"ρ_r /ρ_crit0 = {rad_cmb:.4e}")
    print(f"ρ_de / ρ_m = {rde_cmb/rm_cmb:.4e}")
    print(f"ρ_de / ρ_r = {rde_cmb/rad_cmb:.4e}")
    print("\nPlanck 观测要求暗能量在早期可以忽略 (Ω_de << 0.01)，")
    print("若 ρ_de 在 CMB 时超过辐射密度，则 CMB 声学峰将被彻底破坏。")
    print("=" * 80)

if __name__ == "__main__":
    main()