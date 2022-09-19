import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import sympy as sp
from scipy.integrate import odeint

def SystemOfEquatioons(y, t, R, M, m, g, l, c, alpha, b):
    #y[0] = ksi, y[1] = phi, y[2] = ksi', y[3] = phi'
    #yt[0] = ksi', y[1] = phi', y[2] = ksi'', y[3] = phi''
    d = R * np.cos(alpha)
    thetta = y[0] + y[1] + alpha

    yt = np.zeros_like(y)
    yt[0] = y[2]
    yt[1] = y[3]

    a11 = (M+2*m) * R**2 + M*(d**2 + (l**2 / 12) + 2*R*d*np.cos(thetta))
    a12 = M * (d**2 + (l**2 / 12) + R*d*np.cos(thetta))
    a21 = M * (d**2 + (l**2 / 12) + R*d*np.cos(thetta))
    a22 = M * (d**2 + (l**2 / 12))

    b1 = M*d*(R * (y[2] + y[3])**2 + g)*np.sin(thetta)
    b2 = M*g*d*np.sin(thetta) - c*(2 * R * np.sin(y[1] / 2) - b) * R * np.cos(y[1] / 2)

    yt[2] = (b1*a22 - a12*b2) / (a11*a22 - a12*a21)
    yt[3] = (b1*a11 - a21*b1) / (a11*a22 - a12*a21)

    return yt

def RXRY(Radius, M, m, d, g, Dksi, DDksi, Dphi, DDphi, thetta):
    Dksi = np.array(Dksi)
    DDksi = np.array(DDksi)
    Dphi = np.array(Dphi)
    DDphi = np.array(DDphi)
    Rx = -(M + m)*Radius*DDksi - M*d*((DDksi + DDphi) * np.cos(thetta) - (Dksi + Dphi)**2 * np.sin(thetta))
    Ry = -(M + m)*g - M*d*((DDksi + DDphi) * np.cos(thetta) - (Dksi + Dphi)**2 * np.sin(thetta))
    return Rx, Ry

Radius = 3
M = 150
m = 15
g = 9.81
l = 5
c = 40
alpha = np.pi/6
b = 1
d = Radius * np.cos(alpha)

Ksi0 = 1
Phi0 = 1
DKsi0 = 1
DPhi0 = 1
y0 = [Ksi0, Phi0, DKsi0, DPhi0]
Tfin = 10
NT = 1001
t = np.linspace(0, Tfin, NT)
Y = odeint(SystemOfEquatioons, y0, t, (Radius, M, m, g, l, c, alpha, b))

ksi = Y[:, 0]
phi = Y[:, 1]
Dksi = Y[:, 2]
Dphi = Y[:, 3]
DDksi = [SystemOfEquatioons(y, t, Radius, M, m, g, l, c, alpha, b)[2] for y, t in zip(Y, t)]
DDphi = [SystemOfEquatioons(y, t, Radius, M, m, g, l, c, alpha, b)[3] for y, t in zip(Y, t)]

thetta = Y[:, 0] + Y[:, 1] + alpha
RXRY(Radius, M, m, d, g, Dksi, DDksi, Dphi, DDphi, thetta)
Rx, Ry = RXRY(Radius, M, m, d, g, Dksi, DDksi, Dphi, DDphi, thetta)


fig_for_graphs = plt.figure(figsize=[13,7])
ax_for_graphs = fig_for_graphs.add_subplot(2,3,1)
ax_for_graphs.plot(t,ksi,color='blue')
ax_for_graphs.set_title("ksi(t)")
ax_for_graphs.set(xlim=[0,Tfin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2,3,2)
ax_for_graphs.plot(t,phi,color='red')
ax_for_graphs.set_title("phi(t)")
ax_for_graphs.set(xlim=[0,Tfin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2,2,3)
ax_for_graphs.plot(t,Rx,color='green')
ax_for_graphs.set_title("Rx(t)")
ax_for_graphs.set(xlim=[0,Tfin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2,2,4)
ax_for_graphs.plot(t,Ry,color='black')
ax_for_graphs.set_title("Ry(t)")
ax_for_graphs.set(xlim=[0,Tfin])
ax_for_graphs.grid(True)

#Точка O
X_O = Radius + ksi
Y_O = Radius

X_D = X_O - Radius * np.sin(ksi - 0.3)  # точка D по оси х
Y_D = Y_O - Radius * np.cos(ksi - 0.3)

X_B = X_O + Radius * np.sin(ksi + 0.3)  # точка В по оси х
Y_B = Y_O + Radius * np.cos(ksi + 0.3)

#Круг вокруг О
psi = np.linspace(0, 6.28, 200)
X_circle = Radius*np.sin(psi)
Y_circle = Radius*np.cos(psi)

alpha = ksi/Radius

#диаметр просто так
X_D1 = np.array([X_O + Radius * np.sin(alpha), X_O - Radius * np.sin(alpha)])
Y_D1 = np.array([Y_O + Radius * np.cos(alpha), Y_O - Radius * np.cos(alpha)])

X_A = X_O + Radius * np.sin(alpha + 0.3)
Y_A = Y_O + Radius * np.cos(alpha + 0.3)

#оси
X_Ground = [0, 0, 9]
Y_Ground = [9, 0, 0]

# настройки холста
fig = plt.figure(figsize = [10, 9])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[-1, 10], ylim=[-1, 10])

ax.plot(X_Ground, Y_Ground, color = 'black', linewidth = 3) # система координат
Point_O = ax.plot(X_O[0], Y_O, marker = 'o')[0] # точка А
Drawed_Wheel1 = ax.plot(X_O[0] + X_circle, Y_O + Y_circle)[0] #круг
Drawed_WheelD1 = ax.plot([X_O[0] + Radius * np.sin(alpha[0]), X_O[0] - Radius * np.sin(alpha[0])],
                         [Y_O + Radius * np.cos(alpha[0]), Y_O - Radius * np.cos(alpha[0])])[0] # симуляция кручения колеса 1
Point_D = ax.plot(X_D[0], Y_D[0], marker = 'o', markersize = 10)[0] # точка D
Point_B = ax.plot(X_B[0], Y_B[0], marker = 'o', markersize = 10)[0] # точка B
Line_BD = ax.plot([X_D[0], X_B[0]], [Y_D[0], Y_B[0]])[0] #линия BD
Point_A = ax.plot(X_A[0], Y_A[0], marker = 'o', markersize = 10)[0] # точка А
Spr_AB = ax.plot([X_A[0], X_B[0]], [Y_A[0], Y_B[0]], ls=':')[0] #пружинка AB

def Kino(i):
    Point_O.set_data(X_O[i], Y_O)
    Drawed_Wheel1.set_data(X_O[i] + X_circle, Y_O + Y_circle)
    Drawed_WheelD1.set_data([X_O[i] + Radius * np.sin(alpha[i]), X_O[i] - Radius * np.sin(alpha[i])],
                            [Y_O + Radius * np.cos(alpha[i]), Y_O - Radius * np.cos(alpha[i])])
    Point_D.set_data(X_D[i], Y_D[i])
    Point_B.set_data(X_B[i], Y_B[i])
    Line_BD.set_data([X_D[i], X_B[i]], [Y_D[i], Y_B[i]])
    Point_A.set_data(X_A[i], Y_A[i])
    Spr_AB.set_data([X_A[i], X_B[i]], [Y_A[i], Y_B[i]])

    return np.array([Point_O, Drawed_Wheel1, Drawed_WheelD1, Point_D, Point_B, Line_BD, Spr_AB])

out = FuncAnimation(fig, Kino, frames=NT, interval=10)

plt.show()
