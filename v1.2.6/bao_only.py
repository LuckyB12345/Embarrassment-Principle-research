#!/usr/bin/env python3
"""
BAO-only fit with your OscillatingDETheory (uses Cobaya backend, but no yaml)
"""

from cobaya.theory import Theory
import numpy as np
from scipy.integrate import quad

class OscillatingDETheory(Theory):
    params = {
        "A": {"prior": {"min": 0, "max": 10}, "ref": 1.5},
        "b": {"prior": {"min": 10, "max": 50}, "ref": 26.0},
        "omegam": {"prior": {"min": 0.01, "max": 0.5}, "ref": 0.3},
        "H0": {"prior": {"min": 65, "max": 78}, "ref": 70},
    }

    def initialize(self):
        self.c_light = 299792.458  # km/s

    def get_requirements(self):
        # 声明该理论类可以提供的量
        return {
            "angular_diameter_distance": {"z": "z"},
            "comoving_radial_distance": {"z": "z"},
            "Hubble": {"z": "z"},
        }

    def _w(self, z):
        return -1.0 + self.A * np.sin(self.b * z) / (1+z)**2

    def _de_density_scale(self, z):
        if z == 0:
            return 1.0
        def integrand(zp):
            return 3 * (self._w(zp) + 1) / (1+zp)
        res, _ = quad(integrand, 0, z, epsrel=1e-7, limit=500)
        return np.exp(res)

    def _H(self, z):
        rho_de = self._de_density_scale(z)
        return self.H0 * np.sqrt(self.omegam*(1+z)**3 + (1-self.omegam)*rho_de)

    def _comoving_distance(self, z):
        if z == 0:
            return 0.0
        def integrand(zp):
            return self.c_light / self._H(zp)
        res, _ = quad(integrand, 0, z, epsrel=1e-7, limit=500)
        return res

    # 以下是 Cobaya 要求的提供者方法（方法名必须与 get_requirements 中一致）
    def angular_diameter_distance(self, z):
        """返回角直径距离 D_A(z) [Mpc]"""
        d_c = self._comoving_distance(z)
        return d_c / (1 + np.atleast_1d(z))

    def comoving_radial_distance(self, z):
        """返回共动径向距离 [Mpc]"""
        return self._comoving_distance(z)

    def Hubble(self, z):
        """返回哈勃参数 H(z) [km/s/Mpc]"""
        return self._H(z)

    def calculate(self, state, want_derived=False, **params):
        # 将参数存储到实例变量，供提供者方法使用
        self.A = params['A']
        self.b = params['b']
        self.omegam = params['omegam']
        self.H0 = params['H0']
        if want_derived:
            if 'derived' not in state:
                state['derived'] = {}
            state['derived']['rdrag'] = 147.09  # 声视界尺度
        return True
# Cobaya 配置
info = {
    "packages_path": r"D:\cobaya_data",
    "params": {
        "A": {"prior": {"min": 0, "max": 10}, "ref": 1.5},
        "b": {"prior": {"min": 10, "max": 50}, "ref": 26.0},
        "omegam": {"prior": {"min": 0.01, "max": 0.5}, "ref": 0.3},
        "H0": {"prior": {"min": 65, "max": 78}, "ref": 70},
    },
    "likelihood": {"bao.desi_dr2": None},
    "theory": {"mytheory": {"class": OscillatingDETheory}},
    "output": "chains/bao_only_test",
    "sampler": {"mcmc": {"max_samples": 3000, "burn_in": 1000}},
    "force": True,
    "debug": True,
}

if __name__ == "__main__":
    print("Running BAO-only with your oscillating model...")
    updated_info, sampler = run(info)
    best = sampler.products()["bestfit"]
    print("Best-fit:", best)