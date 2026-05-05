import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ========== 常量 ==========
C_LIGHT = 299792.458      # km/s
RD_FID = 147.09           # Mpc
INTEGRAL_EPSREL = 1e-7    # 高精度（如太慢可改为1e-5）
INTEGRAL_LIMIT = 500

# ========== 最终最佳拟合参数（来自联合拟合，公平比较）==========
A = 1.5355      # 振荡振幅
b = 26.42       # 振荡频率
Om = 0.1252     # 物质密度参数
H0 = 69.90      # 哈勃常数 km/s/Mpc

# ========== 模型函数 ==========
def one_plus_w(z):
    """1 + w(z)"""
    return A * np.sin(b * z) / (1+z)**2

def rho_osc_at_z(z):
    """暗能量密度相对演化 ρ_de(z)/ρ_de0"""
    def integrand(x):
        return 3 * one_plus_w(x) / (1+x)
    val, _ = quad(integrand, 0, z, epsrel=INTEGRAL_EPSREL, limit=INTEGRAL_LIMIT)
    val = np.clip(val, -50, 50)   # 防止指数溢出
    return np.exp(val)

def H(z):
    """哈勃参数 H(z) [km/s/Mpc]"""
    rho = rho_osc_at_z(z)
    rho = np.clip(rho, 1e-30, 1e30)
    return H0 * np.sqrt(Om*(1+z)**3 + (1-Om)*rho)

def mu_osc(z):
    """距离模量 μ(z)"""
    def integrand(x):
        rho = rho_osc_at_z(x)
        rho = np.clip(rho, 1e-30, 1e30)
        return 1.0 / np.sqrt(Om*(1+x)**3 + (1-Om)*rho)
    I, _ = quad(integrand, 0, z, epsrel=INTEGRAL_EPSREL, limit=INTEGRAL_LIMIT)
    dc = (C_LIGHT / H0) * I          # 共动距离 [Mpc]
    return 5 * np.log10(dc * (1+z)) + 25

# ========== 生成数据 ==========
z_plot = np.linspace(0, 1.0, 300)
wp = one_plus_w(z_plot)
Hz = [H(zi) for zi in z_plot]

# ========== 图1：1+w(z) ==========
plt.figure(figsize=(6,3))
plt.plot(z_plot, wp, 'r-', lw=2)
plt.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.xlabel(r'$z$')
plt.ylabel(r'$1+w(z)$')
plt.title(r'$1+w(z) = A \sin(bz)/(1+z)^2$')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('fig_wz_final.png', dpi=150)
plt.close()

# ========== 图2：H(z) ==========
plt.figure(figsize=(6,3))
plt.plot(z_plot, Hz, 'r-', lw=2)
plt.xlabel(r'$z$')
plt.ylabel(r'$H(z)$ [km/s/Mpc]')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('fig_Hz_final.png', dpi=150)
plt.close()

# ========== 图3：Pantheon+ 残差 ==========
data = np.genfromtxt("pantheon+_data.txt", skip_header=1, usecols=(4,10,11))
z_data, mu_data, mu_err = data.T
mu_model = np.array([mu_osc(z) for z in z_data])
res = mu_data - mu_model

plt.figure(figsize=(6,3))
plt.errorbar(z_data, res, yerr=mu_err, fmt='k.', ms=2, alpha=0.6, capsize=0)
plt.axhline(0, color='r', lw=2)
plt.xlabel(r'$z$')
plt.ylabel(r'$\mu_{\rm obs} - \mu_{\rm model}$')
plt.title('Pantheon+ Residuals')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('fig_residual_final.png', dpi=150)
plt.close()

print("三张图已生成，使用最终联合拟合最佳拟合参数：")
print(f"A = {A:.4f}, b = {b:.4f}, Ωm = {Om:.4f}, H0 = {H0:.2f}")