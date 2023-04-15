from typing import Callable
import numpy as np
from dual import Dual

#all based on this:
#https://www.youtube.com/watch?v=ceaNqdHdqtg&ab_channel=MichaelPenn

def dual_diff_scalars(f: Callable, x: float):
    """
    f - function: scalar -> scalar
    x - scalar variable
    """

    # Function value at x
    fx = f(x)

    # Derivative at x
    Dx = x + Dual(0,1)
    tmp : Dual = f(Dx) - fx
    dfdx = tmp.dual

    return fx, dfdx


def dual_diff_jacobian(f: Callable, x: np.ndarray):
    """
    f - function: vector -> vector
    x - vector
    """
    # Function value at x
    fx = f(x)

    n = len(fx)
    m = len(x)

    jacobian = np.zeros((m, n))
    for i in range(n):
        for j in range(m):
            fi = lambda x: f(x)[i] #create a new functoin that only returns the ith component of f(x)
            Dxj = x + Dual(0,0)
            Dxj[j] = x[j] + Dual(0,1) #this is the variable we are taking the derivative with respect to
            
            tmp : Dual = fi(Dxj) - fx[i]
            jacobian[j,i] = tmp.dual

    return fx, jacobian


