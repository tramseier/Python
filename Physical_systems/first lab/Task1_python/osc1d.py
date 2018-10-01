"""Solve the one dimensional harmonic oscillator
d2x/dt2 = -f^2*x-c*v, x[0] = 0, v[0] = 1
w: frequency
c: damping coefficient
x: displacement
v: velocity
t: time

Wei Chen, 2011-2013 
"""

import numpy as np
import matplotlib.pylab as plt

def exact(t,w):
    # exact solution based on the initial value: 
    x_ex = np.sin(t)/w
    v_ex = np.cos(t)
    return x_ex, v_ex

def euler(x,v,dt,w,c,corr):
    # d2x/dt2 = -w^2*x-c*v
    # x(i+1)-x(i) = dt*v
    # switch to pred/corr scheme if corr == True
    dt /= w
    for i in xrange(x.size-1):
        x[i+1] = x[i] + dt * v[i]
        v[i+1] = v[i] - dt * (w**2*x[i] + c*v[i])
        if corr == True:
            x[i+1] = x[i] + dt * (v[i] + v[i+1])/2
            v[i+1] = v[i] - dt * (w**2*(x[i] + x[i+1])/2 + c*(v[i] + v[i+1])/2)
    return x, v

def euler2nd(x,v,dt,w,c):
    # include 2nd order term in x[i] using Euler method
    # x[i+1] = x[i] + dt * v[i] + (dt)**2/2*a[i]
    dt /= w
    for i in xrange(x.size-1):
        x[i+1] = x[i] + dt * v[i] - (dt)**2/2 * x[i]
        v[i+1] = v[i] - dt * (w**2*x[i] + c*v[i])
    return x, v

def verlet(x,v,dt,w,c):
    # Verlet method
    # x(i+2) = 2x(i+1) - x(i) + (dt)^2*a
    # v(i+1) = [x(i+2) - x(i)] / (2*dt)
    dt /= w
    x[1] = x[0] + dt * v[0]
    for i in xrange(x.size-2):
        x[i+2] = 2*x[i+1] - x[i] - (dt)**2*(w**2*x[i] + c*v[i])
        v[i+1] = (x[i+2] - x[i]) / (2*dt)
    return x, v

def verletv(x,v,dt,w,c):
    # Velocity Verlet algorithm
    # x[i+1] = x[i] + dt*v[i] + (dt)**2/2*a[i]
    # v[i+1] = v[i] -dt/2*(a[i]+a[i+1])
    dt /= w
    for i in xrange(x.size-1):
        x[i+1] = x[i] + dt * v[i] - (dt)**2/2*(w**2*x[i] + c*v[i])
        v[i+1] = v[i] - dt/2*(w**2*(x[i+1]+ x[i])+c*(v[i+1]+v[i]))
    return x, v

def plotxv(x_ex, v_ex, x, v, t):
    font = {'fontname':'sans-serif'}
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, x_ex, 'r-', t, x, 'r.', t, v_ex, 'g-', t, v, 'g.')
    ax.set_xlabel('ft', **font)  
    ax.autoscale(axis='x', tight=True)
    plt.show()
 
def plotE(E, t):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, E, 'b.')
    ax.set_xlabel('ft')
    ax.autoscale(axis='x', tight=True)
    plt.show()

if __name__ == "__main__":
    # dt: step
    # x[0], v[0]: boundary conditions
    # j: number of cycles
    w = 1; c = 0.0; n0 = 500; j = 5
    n = n0*j
    dt = 2*np.pi/n0     
    t = np.zeros((n,))  # t --> w*t
    for i in xrange(n):
        t[i] = i*dt
    x = np.zeros((n,)); v = np.zeros((n,))
    x[0] = 0; v[0] = 1
    X_ex, V_ex = exact(t)
    X, V = euler(x,v,dt,w,c,corr=False)
    #X, V = euler2nd(x,v,dt,w,c)
    #X, V = verletv(x,v,dt,w,c)
    E = x**2 + v**2
    plotxv(X_ex, V_ex, X, V, t)
    plotE(E,t)
