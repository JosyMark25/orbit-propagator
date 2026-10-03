import math

def solve_kepler(M, e, maxsteps, presicion=1e-9):
    if maxsteps < 1:
        raise ValueError(f"maxsteps 必须 >= 1，收到 {maxsteps}")
    if(M < math.pi):
        E0 = M + e / 2
    else:
        E0 = M - e / 2
    for j in range(maxsteps):
        func = E0 - e * math.sin(E0) - M
        if(abs(func) < presicion):
            return E0
        derived_func = 1 - e * math.cos(E0)
        E0 = E0 - func / derived_func
    if (abs(func) >= presicion):
        raise RuntimeError(f"Kepler未收敛, M = {M:.9f}, e = {e}, maxsteps = {maxsteps} , 残差为func{abs(func):.8f}, 可以查询e是否在定义域或者增大maxsteps")

    return E0

def tran_E2nju(E, e):
    return 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2),
                          math.sqrt(1 - e) * math.cos(E / 2))
    
def tran_nju2E(nu, e):
    return 2 * math.atan2(math.sqrt(1 - e) * math.sin(nu / 2),
                          math.sqrt(1 + e) * math.cos(nu / 2))

'''def M_to_E(M, e):
    ...

def E_to_M(E, e):
    M = E - e * math.sin(E)
    return M '''