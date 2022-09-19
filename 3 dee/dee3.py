import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import sympy as sp
from scipy.integrate import odeint

def SystemOfEquatioons(y, t, g, c, l, Mu, P):
    #y[0] = s, y[1] = phi, y[2] = s', y[3] = phi'
    #yt[0] = s', y[1] = phi', y[2] = s'', y[3] = phi''

    yt = np.zeros_like(y)
    yt[0] = y[2]
    yt[1] = y[3]

    a11 = 1#s''
    a12 = 0#phi''
    a21 = 0#s''
    a22 = l + y[0]#phi''

    b1 = -1 * ((Mu*g/P) * y[2] +(c*g/P) * y[0] - (l+y[0]) * (y[3]**2) - g * np.cos(y[1]))
    b2 = -1 * (y[2] *  y[3] + g*(y[0]+l)*np.sin(y[1]))

    yt[2] = (b1*a22 - a12*b2) / (a11*a22 - a12*a21)
    yt[3] = (b1*a11 - a21*b1) / (a11*a22 - a12*a21)

    return yt

def N(g, l, P, s, phi, Ds, DDs, Dphi, DDphi):
    s = np.array(s)
    Ds = np.array(Ds)
    DDs = np.array(DDs)
    phi = np.array(phi)
    Dphi = np.array(Dphi)
    DDphi = np.array(DDphi)
    N = P * np.sin(phi) + (P/g) * ((l + s)*DDphi + 2*Dphi*Ds)
    return N
#variables
g = 9.81
c = 10
l = 3
Mu = 1
P = 100
P1 = 50
L = 10

S0 = 0
Phi0 = - 1/2 * np.pi#start
DS0 = 1
DPhi0 = 3/4
y0 = [S0, Phi0, DS0, DPhi0]

Tfin = 1/4 * np.pi
NT = 500
t = np.linspace(- 1/2 * np.pi, Tfin, NT)
Y = odeint(SystemOfEquatioons, y0, t, (g, c, l, Mu, P))

s = Y[:, 0]
phi = Y[:, 1]
Ds = Y[:, 2]
Dphi = Y[:, 3]
DDs = [SystemOfEquatioons(y, t, g, c, l, Mu, P)[2] for y, t in zip(Y, t)]
DDphi = [SystemOfEquatioons(y, t, g, c, l, Mu, P)[3] for y, t in zip(Y, t)]

N = N(g, l, P, s, phi, Ds, DDs, Dphi, DDphi)

fig_for_graphs = plt.figure(figsize=[13,7])
ax_for_graphs = fig_for_graphs.add_subplot(3,1,1)
ax_for_graphs.plot(t,s,color='blue')
ax_for_graphs.set_title("s(t)")
ax_for_graphs.set(xlim=[0,Tfin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(3,1,2)
ax_for_graphs.plot(t,phi,color='red')
ax_for_graphs.set_title("phi(t)")
ax_for_graphs.set(xlim=[0,Tfin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(3,1,3)
ax_for_graphs.plot(t,N,color='green')
ax_for_graphs.set_title("N(t)")
ax_for_graphs.set(xlim=[- 1/2 * np.pi,Tfin])
ax_for_graphs.grid(True)



#Point O
X_O = 3
Y_O = 6

# point M
X_M = l * np.cos(phi) + X_O
Y_M = l * np.sin(phi) + Y_O

X_cir = np.abs(np.sin(phi)) * l * np.cos(phi) + X_O
Y_cir = np.abs(np.sin(phi)) * l * np.sin(phi) + Y_O
#axis
X_Ground = [0, 0, 9]
Y_Ground = [9, 0, 0]

# plotting
fig = plt.figure(figsize = [10, 9])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[-1, 10], ylim=[-1, 10])

ax.plot(X_Ground, Y_Ground, color = 'black', linewidth = 3) # система координат
Point_O = ax.plot(X_O, Y_O, marker = '.')[0] # точка O
Point_M = ax.plot(X_M, Y_M, marker = '.')[0] # точка M
Line_OM = ax.plot([X_O, X_M[0]], [Y_O, Y_M[0]], linewidth=5)[0] #линия АВ
Point_cir = ax.plot(X_cir, Y_cir, marker = 'o')[0]
Spr_Ocir = ax.plot([X_O, X_cir[0]], [Y_O, Y_cir[0]], linestyle=':', linewidth=3)[0]

def Kino(i):
    Point_O.set_data(X_O, Y_O)
    Point_M.set_data(X_M[i], Y_M[i])
    Line_OM.set_data([X_O, X_M[i]], [Y_O, Y_M[i]])
    Point_cir.set_data(X_cir[i], Y_cir[i])
    Spr_Ocir.set_data([X_O, X_cir[i]], [Y_O, Y_cir[i]])
    return np.array([Point_O, Point_M, Line_OM, Point_cir, Spr_Ocir])

out = FuncAnimation(fig, Kino, frames=NT, interval=10)

plt.show()
