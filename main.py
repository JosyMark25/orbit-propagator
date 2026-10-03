from constants import mju_earth
from propagate import propagate_from_coe
from visualize import plot_orbit_3d
import numpy as np
import math

def main():
    maxsteps = int(input())

    print("按 a, e, i, Ω, ω, ν 输入：")
    vec = np.array(input().split(), dtype=float)
    a, e, i, Omega, omega, nu0 = vec

    i = math.radians(i)
    Omega = math.radians(Omega)
    omega = math.radians(omega)
    nu0 = math.radians(nu0)
    
    T = 2 * np.pi * np.sqrt(a**3 / mju_earth)
    t_array = np.linspace(0, T, maxsteps)
    
    r_array, v_array = propagate_from_coe(a, e, i, Omega, omega, nu0, t_array, maxsteps, mju_earth)
    
    plot_orbit_3d(r_array, save_path='/home/osyark/orbit_V1/orbit.png')

if __name__ == '__main__':
    main()