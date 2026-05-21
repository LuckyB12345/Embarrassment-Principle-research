"""
验证 BAO 代码：固定 Planck 2018 宇宙学参数，
计算每个红移点的理论 D_V/r_d，与修正后的 DESI DR2 观测值对比。
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM

# 常量（与您代码中一致）
C_LIGHT = 299792.458   # km/s
RD_FID = 147.09        # Mpc

def bao_D_V(cosmo, z):
    """计算 D_V / r_d"""
    d_c = cosmo.comoving_distance(z).value   # Mpc
    H_z = cosmo.H(z).value                   # km/s/Mpc
    term = (d_c**2) * (C_LIGHT * z / H_z)
    D_V = term ** (1/3)                     # Mpc
    return D_V / RD_FID

# 固定标准宇宙学参数 (Planck 2018)
cosmo_planck = FlatLambdaCDM(H0=67.66, Om0=0.3111)

# 修正后的 DESI DR2 BAO 观测数据 (D_V/r_d, 误差)
bao_obs = {
    0.295: (7.391, 0.072),
    0.51:  (13.767, 0.061),
    0.70:  (18.651, 0.071),
    0.93:  (22.616, 0.068),
    1.32:  (27.660, 0.110),
    1.48:  (29.550, 0.150),
}

print("验证 BAO 代码：固定 ΛCDM 参数 (H0=67.66, Ωm=0.3111)")
print("红移   理论值    观测值    偏差 (sigma)")
print("-" * 50)

all_sigma = []
for z, (obs, err) in bao_obs.items():
    pred = bao_D_V(cosmo_planck, z)
    sigma = (pred - obs) / err
    all_sigma.append(abs(sigma))
    print(f"{z:.3f}   {pred:.4f}   {obs:.4f}   {sigma:+.2f}")

print("-" * 50)
print(f"最大残差 sigma = {max(all_sigma):.2f}")
print("如果所有 |sigma| < 3，说明您的 BAO 计算正确（考虑 DESI-Planck 约 2.3σ 张力）。")