#!/bin/env python

import osc1d as solver
import numpy as np

# dt: step
# x[0], v[0]: boundary conditions
# j: number of cycles
w = 1; c = 1.0; n0 = 500; j = 5
n = n0*j
dt = 2*np.pi/n0
t = np.zeros((n,))  # t --> w*t
for i in xrange(n):
    t[i] = i*dt
x = np.zeros((n,)); v = np.zeros((n,))
x[0] = 0; v[0] = 1

X_ex, V_ex = solver.exact(t,w)
X, V = solver.euler(x,v,dt,w,c,corr=False)
X, V = solver.euler2nd(x,v,dt,w,c)
X, V = solver.verletv(x,v,dt,w,c)

maxd = np.max(np.abs(X-X_ex))  # max deviation of X

E = x**2 + v**2
poly = np.polyfit(t,E,1)
g_E = lambda x: poly[0]*x + poly[1]
E_t = g_E(t)  # fitted E

rms = np.sqrt(1.0/(n-2)*np.sum((E-E_t)**2)) # r.m.s. of deviation 

print("max deviation of the displacement: {}".format(maxd))
print("drift of the energy: {}".format(poly[0]))
print("rms of the deviation of the energy: {}".format(rms))

solver.plotxv(X_ex, V_ex, X, V, t)
solver.plotE(E,t)
