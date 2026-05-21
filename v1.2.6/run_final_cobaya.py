import numpy as np
from astropy.cosmology import FlatLambdaCDM

C_LIGHT = 299792.458   # km/s
RD_FID = 147.09        # Mpc

def bao_D_V(cosmo, z):
    d_c = cosmo.comoving_distance(z).value   # Mpc
    H_z = cosmo.H(z).value                   # km/s/Mpc
    term = (d_c**2) * (C_LIGHT * z / H_z)
    D_V = term ** (1/3)                      # Mpc
    return D_V / RD_FID

# 使用 Planck 2018 最佳拟合参数
cosmo_planck = FlatLambdaCDM(H0=67.7, Om0=0.311)

# DESI DR2 的观测值 (D_V/r_d)
obs = {
    0.20: 3.861,
    0.51: 13.767,
    0.70: 18.651,
    0.93: 22.616,
    1.32: 27.660,
    1.48: 29.550,
}

print("理论预测 vs 观测值 (ΛCDM Planck 参数):")
for z, ob in obs.items():
    pred = bao_D_V(cosmo_planck, z)
    # 粗略估计 sigma（仅用于趋势判断，非精确）
    err = {
        0.20: 0.045, 0.51: 0.061, 0.70: 0.071,
        0.93: 0.068, 1.32: 0.110, 1.48: 0.150
    }[z]
    sigma = (pred - ob) / err
    print(f"z={z:.2f}: pred={pred:.4f}, obs={ob:.4f}, sigma={sigma:.1f}")