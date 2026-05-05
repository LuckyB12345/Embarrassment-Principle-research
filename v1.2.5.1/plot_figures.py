import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

C_LIGHT = 299792.458
RD_FID = 147.09
INTEGRAL_EPSREL = 1e-4
INTEGRAL_LIMIT = 500

A    = 1.4087
b    = 26.60
Om   = 0.2691
H0   = 70.16

def one_plus_w(z):
    return A * np.sin(b * z) / (1+z)**2

def rho_osc_at_z(z):
    def integrand(x): return 3 * one_plus_w(x) / (1+x)
    val, _ = quad(integrand, 0, z, epsrel=INTEGRAL_EPSREL, limit=INTEGRAL_LIMIT)
    val = np.clip(val, -50, 50)
    return np.exp(val)

def H(z):
    rho = rho_osc_at_z(z)
    rho = np.clip(rho, 1e-30, 1e30)
    return H0 * np.sqrt(Om*(1+z)**3 + (1-Om)*rho)

def mu_osc(z):
    def integrand(x):
        rho = rho_osc_at_z(x)
        rho = np.clip(rho, 1e-30, 1e30)
        return 1.0 / np.sqrt(Om*(1+x)**3 + (1-Om)*rho)
    I, _ = quad(integrand, 0, z, epsrel=INTEGRAL_EPSREL, limit=INTEGRAL_LIMIT)
    dc = (C_LIGHT / H0) * I
    return 5 * np.log10(dc * (1+z)) + 25

z_plot = np.linspace(0, 1.0, 300)

# 1+w(z)
plt.figure(figsize=(6,3))
plt.plot(z_plot, one_plus_w(z_plot), 'r-', lw=2)
plt.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('z'); plt.ylabel('1+w(z)')
plt.title('1+w(z) = A sin(bz)/(1+z)²')
plt.grid(alpha=0.2); plt.tight_layout()
plt.savefig('fig_wz_final.png', dpi=150); plt.close()

# H(z)
H_vals = np.array([H(z) for z in z_plot])
plt.figure(figsize=(6,3))
plt.plot(z_plot, H_vals, 'r-', lw=2)
plt.xlabel('z'); plt.ylabel('H(z) [km/s/Mpc]')
plt.grid(alpha=0.2); plt.tight_layout()
plt.savefig('fig_Hz_final.png', dpi=150); plt.close()

# 残差
data = np.genfromtxt("pantheon+_data.txt", skip_header=1, usecols=(4,10,11))
z_data, mu_data, mu_err = data.T
mu_model = np.array([mu_osc(z) for z in z_data])
res = mu_data - mu_model
plt.figure(figsize=(6,3))
plt.errorbar(z_data, res, yerr=mu_err, fmt='k.', ms=2, alpha=0.6)
plt.axhline(0, color='r', lw=2)
plt.xlabel('z'); plt.ylabel('μ_{obs} - μ_{model}')
plt.title('Pantheon+ Residuals')
plt.grid(alpha=0.2); plt.tight_layout()
plt.savefig('fig_residual_final.png', dpi=150); plt.close()

print("三张图已生成 (正确距离公式)。")