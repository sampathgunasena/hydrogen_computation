########################################################################################
#
#script that solves schrodinger equation via generalised eigen value decomposition method
#
#######################################################################################

import numpy as np
from scipy import linalg
import math
from matplotlib import pyplot as plt



# We will use atomic units throughout the code
alpha = np.array([13.00773, 1.962079, 0.444529, 0.1219492])
#alpha=np.random.rand(4)

#creating multi-dim arrays to hold values for overlap, kinetic, potential matrices 
S = np.zeros((4,4))
T = np.zeros((4,4))
V = np.zeros((4,4))

#Compute the overlap, kinetic and potential energies matrices
for i in range(len(alpha)):
  for j in range(len(alpha)):
    S[i,j] = (math.pi/(alpha[i]+alpha[j]))**(3./2.)
    T[i,j] = 3.*(alpha[i]*alpha[j]*math.pi**(3./2.))/(alpha[i]+alpha[j])**(5./2.)
    V[i,j] = -2.*math.pi/(alpha[i]+alpha[j])

#creating Hamiltonain matrix
H = T + V


#Solve the generalized eigenvalue problem
# eval and vec are arrays with eigenvuales and eigenvectors
# The eigenvualue corresponding to val[i] is the column vec[:,i].
# eigen value => energy, corresponding eigen vector => c coefficient 
val, vec = linalg.eig(H,S)
print("Eigenvalues: ", val.real)
print("Eigenvectors: ", vec.real)


# Print the ground state energy
print("Ground State Energy: ", val.real.min())
# Index of the ground state eigenvalue and eigenvector
imin = val.real.argmin()
# Ground state eigenvector, i.e. the gaussian coefficients of the ground state wavefunction
vec_ground = np.atleast_1d(vec[:,imin])
print("Ground State Eigenvector: ", vec_ground)


#Normalize the ground state wavefunction. 
norm = 0.0
for i in range(len(alpha)):
  for j in range(len(alpha)):
    norm = norm + vec_ground[i] * vec_ground[j] * S[i,j]

vec_ground = vec_ground / math.sqrt(norm)

print("Normalized eigen vector: ", vec_ground)

#Plot numerical wavefucntion, exact solution, and the individual gaussians.
x = np.linspace(0, 5, 100)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

ax1.plot(x, 1/math.sqrt(np.pi)*np.exp(-x), linewidth=2, color='red', label='Exact')
ax1.plot(x, abs(vec_ground[0]*np.exp(-1.*alpha[0]*x*x) + vec_ground[1]*np.exp(-1.*alpha[1]*x*x) + vec_ground[2]*np.exp(-1.*alpha[2]*x*x) + vec_ground[3]*np.exp(-1.*alpha[3]*x*x)), linewidth=2, color='blue', label='Computational')
ax1.plot(x, abs(vec_ground[0]*np.exp(-1.*alpha[0]*x*x)), linewidth=1, color='black', label=r"$c_1 exp(-\alpha_1r^2)$")
ax1.plot(x, abs(vec_ground[1]*np.exp(-1.*alpha[1]*x*x)), linewidth=1, color='black', label=r"$c_2 exp(-\alpha_2r^2)$")
ax1.plot(x, abs(vec_ground[2]*np.exp(-1.*alpha[2]*x*x)), linewidth=1, color='black', label=r"$c_3 exp(-\alpha_3r^2)$")
ax1.plot(x, abs(vec_ground[3]*np.exp(-1.*alpha[3]*x*x)), linewidth=1, color='black', label=r"$c_4 exp(-\alpha_4r^2)$")
plt.title('Ground State Energy: %.6f hartree' % val.real.min())
plt.xlabel('r (bohr)')
plt.ylabel('1/bohr^3')
plt.legend(loc='upper right')
ax1.text(0.18, 0.6, r"$\Psi = \sum_i c_i exp(-\alpha_ir^2)$", transform=ax1.transAxes, fontsize=14,
        verticalalignment='top')
plt.show()










