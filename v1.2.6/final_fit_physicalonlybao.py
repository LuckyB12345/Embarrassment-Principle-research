#!/usr/bin/env python3
"""
Cobaya 联合拟合: BAO (DESI DR2) + SN (Pantheon+)
模型: 1+w(z) = A sin(bz)/(1+z)^2
不含 Planck 似然, 避免 CMB 理论需求。
"""

import numpy as np
from scipy.integrate import quad
from cobaya.theory import Theory
from cobaya.run import run
import warnings
warnings.filterwarnings("ignore")

class OscillatingDETheory(Theory):
    params = {
        "A": {"prior": {"min": 0, "max": 10}, "ref": 1.5},
        "b": {"prior": {"min": 10, "max": 50}, "ref": 26.0},
        "omegam": {"prior": {"min": 0.01, "max": 0.5}, "ref": 0.3},
        "H0": {"prior": {"min": 65, "max": 78}, "ref": 70},
    }

    def initialize(self):
        self.c_light = 299792.458  # km/s

    def _w(self, z, A, b):
        return -1.0 + A * np.sin(b * z) / (1+z)**2

    def _de_density_scale(self, z, A, b):
        def integrand(zp):
            return 3 * (self._w(zp, A, b) + 1) / (1+zp)
        res, _ = quad(integrand, 0, z, epsrel=1e-7, limit=500)
        return np.exp(res)

    def H(self, z, A, b, omegam, H0):
        if np.isscalar(z):
            rho_de = self._de_density_scale(z, A, b)
            return H0 * np.sqrt(omegam*(1+z)**3 + (1-omegam)*rho_de)
        return np.array([self.H(zi, A, b, omegam, H0) for zi in z])

    def comoving_distance(self, z, A, b, omegam, H0):
        if z == 0:
            return 0.0
        def integrand(zp):
            return self.c_light / self.H(zp, A, b, omegam, H0)
        res, _ = quad(integrand, 0, z, epsrel=1e-7, limit=500)
        return res

    def calculate(self, state, want_derived=False, **params):
        # 获取参数
        self.A = params['A']
        self.b = params['b']
        self.omegam = params['omegam']
        self.H0 = params['H0']

        # 物理约束：z=0.2 处必须加速膨胀 (1+w < 2/3)
        w0p2 = -1.0 + self.A * np.sin(self.b * 0.2) / (1+0.2)**2
        if w0p2 >= -1/3:
            return False

        # 可选边界检查（先验已经限制，但以防数值溢出）
        if not (0 < self.A < 10 and 10 < self.b < 50 and 0.01 < self.omegam < 0.5 and 65 < self.H0 < 78):
            return False

        if want_derived:
            if 'derived' not in state:
                state['derived'] = {}
            state['derived']['rdrag'] = 147.09   # 固定声视界尺度

        return True

    # ----- 提供者方法（供 Cobaya 似然调用）-----
    def get_Hubble(self, z, **kwargs):
        return self.H(z, self.A, self.b, self.omegam, self.H0)

    def get_angular_diameter_distance(self, z, **kwargs):
        d_c = self.comoving_distance(z, self.A, self.b, self.omegam, self.H0)
        return d_c / (1 + z)

    def get_comoving_radial_distance(self, z, **kwargs):
        return self.comoving_distance(z, self.A, self.b, self.omegam, self.H0)

    def get_BAO(self, z, **kwargs):
        z_arr = np.atleast_1d(z)
        d_c = self.comoving_distance(z_arr, self.A, self.b, self.omegam, self.H0)
        H_z = self.H(z_arr, self.A, self.b, self.omegam, self.H0)
        return {"D_M": d_c, "D_H": self.c_light / H_z}

    def get_mu(self, z, **kwargs):
        d_L = (1+z) * self.comoving_distance(z, self.A, self.b, self.omegam, self.H0)
        return 5 * np.log10(d_L) + 25

    def get_rdrag(self, **kwargs):
        return 147.09


# ========== Cobaya 配置 ==========
info = {
    "packages_path": r"D:\cobaya_data",   # 你的数据存放路径
    "params": {
        "A": {"prior": {"min": 0, "max": 10}, "ref": 1.5},
        "b": {"prior": {"min": 10, "max": 50}, "ref": 26.0},
        "omegam": {"prior": {"min": 0.01, "max": 0.5}, "ref": 0.3},
        "H0": {"prior": {"min": 65, "max": 78}, "ref": 70},
    },
    "likelihood": {
        "bao.desi_dr2": None,
        "sn.pantheonplus": None,
    },
    "theory": {"mytheory": {"class": OscillatingDETheory}},
    "output": "chains/bao_sn_joint",
    "sampler": {
        "mcmc": {
            "Rminus1_stop": 0.02,
            "max_samples": 20000,
            "burn_in": 5000,
        }
    },
    "force": True,
    "debug": False,   # 可以改为 True 以查看详细调试信息
}

if __name__ == "__main__":
    print("启动 Cobaya 联合拟合: BAO (DESI DR2) + SN (Pantheon+)")
    print("模型: 1+w(z) = A sin(bz)/(1+z)^2")
    updated_info, sampler = run(info)
    best = sampler.products()["bestfit"]
    print("\n最佳拟合参数 (最大后验):")
    for p in ["A", "b", "omegam", "H0"]:
        print(f"  {p} = {best[p]:.4f}")