import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import sympy as sp
from scipy.integrate import odeint

Steps = 1001
t = np.linspace(1, 11, Steps) # массив времени

x = 2 + np.cos(4.88 * t)
phi = np.sin(t)


Radius = 3
#Точка O
X_O = Radius + x
Y_O = Radius

X_D = X_O - Radius * np.sin(x - 0.3)  # точка D по оси х
Y_D = Y_O - Radius * np.cos(x - 0.3)

X_B = X_O + Radius * np.sin(x + 0.3)  # точка В по оси х
Y_B = Y_O + Radius * np.cos(x + 0.3)

#Круг вокруг О
psi = np.linspace(0, 6.28, 200)
X_circle = Radius*np.sin(psi)
Y_circle = Radius*np.cos(psi)

alpha = x/Radius

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

out = FuncAnimation(fig, Kino, frames=Steps, interval=10)

plt.show()
