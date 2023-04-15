import numpy as np
from dual import Dual
from dual_diff import dual_diff_scalars, dual_diff_jacobian

#multiplication
a = Dual(2,3)
b = Dual(5,-1)
assert a*b == Dual(10, 13)

#division
a = Dual(3,2)
b = Dual(2,5)
assert a/b == Dual(3/2, -11/4)

#test numpy compatibility
a = Dual(3,2)
b = Dual(2,5)
c = np.array([a,b])
assert np.sum(c) == a + b

#binary vector to epsilon vector
a = np.array([1,0,1])
b = a.astype(Dual) * Dual(0,1) + Dual(1,0)
assert np.array_equal(b,np.array([Dual(1,1),Dual(1,0),Dual(1,1)]))


#test dual_diff_scalars
def f(x : float) -> float:
    return x**2 
x = 3
fx, dfdx = dual_diff_scalars(f,x)
assert fx == 9
assert dfdx == 6

#test dual_diff_jacobian
def f(x : np.ndarray) -> np.ndarray:
    f1 = x[0]**2 + x[1]**2
    f2 = x[0]**3 + x[1]**3
    return np.array([f1,f2])
x = np.array([3.0,4.0])
fx, dfdx = dual_diff_jacobian(f,x)
assert np.array_equal(fx,np.array([25, 91]))
assert np.array_equal(dfdx,np.array([[6,27],[8,48]]))

