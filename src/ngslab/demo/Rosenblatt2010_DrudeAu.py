# Plot and write dispersive refractive indices of gold,
# using Eq. (1) in Ref. [1].
# [1] G. Rosenblatt, E. Feigenbaum, and M. Orenstein, ‘Circular motion of electromagnetic power shaping the dispersion of Surface Plasmon Polaritons’, Opt. Express, OE, vol. 18, no. 25, pp. 25861–25872, Dec. 2010, doi: 10.1364/OE.18.025861.

import numpy as np
import matplotlib.pyplot as plt

# Drude model parameters for gold
omega_p = 1.37e16
gamma = 4.05e13

def DrudeAu(omega):
    epsilon = 1-omega_p**2/(omega*(omega+1j*gamma))
    return epsilon+0j # forced to be complex-valued

omega_min = 0.01*omega_p
omega_max = 2.4*omega_p
npoints=200
omega = np.linspace(omega_min, omega_max, npoints)
ld = 2*np.pi*299792458/omega
epsilon2 = 1
omega_spp = omega_p/np.sqrt(1+epsilon2)
ld_spp = 2*np.pi*299792458/omega_spp
ld_p = 2*np.pi*299792458/omega_p
epsilon = DrudeAu(omega)
n = (epsilon**.5).real
k = (epsilon**.5).imag

# save to a file
file = open('auDrude.ref', 'w')
for i in range(npoints-1, -1, -1):
    file.write('{:.4e} {:.4e} {:.4e} \n'.format(ld[i], n[i], k[i]))
file.close()

# plot n,k vs μm
#
plt.rc('font', family='Arial', size='14')
#
plt.figure(3)
plt.plot(ld, n, label="n")
plt.plot(ld, k, label="k")
plt.xlabel('Wavelength (μm)')
plt.ylabel('n, k')
# plt.xscale('log')
# plt.yscale('log')
plt.legend(bbox_to_anchor=(0,1.02,1,0),loc=3,ncol=2,borderaxespad=0)
