from geo import SE3_Exp, SE3_to_storage
from symforce import geo as sf_geo
from dual_diff import dual_diff_jacobian
import numpy as np

tangent = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

#test diffrentiation
def f(tangent):
    #x - tangnet vector
    X = SE3_Exp(tangent)
    return SE3_to_storage(X)
strg_dual, J_dual = dual_diff_jacobian(f, tangent)

X_sf = sf_geo.Pose3_SE3.from_tangent(tangent)
strg_sf = np.array(X_sf.to_storage(),dtype = float)
J_sf = X_sf.storage_D_tangent().to_numpy()

print(np.linalg.norm(strg_sf - strg_dual) / np.linalg.norm(strg_sf))
print(np.linalg.norm(J_sf.T - J_dual) / np.linalg.norm(J_sf)) #<---------------this is a problem




