import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

stp = 500 #amount of steps
#Точка O
X_O = 3
Y_O = 6

l = 5 # длина палки
phi = 1/4 * np.pi
phi = np.concatenate((np.linspace( - 1/2 * np.pi, 0 - phi, int(stp/2)),
np.linspace( 0 - phi, - 1/2 * np.pi, int(stp/2))), axis=None)
# point M
X_M = l * np.cos(phi) + X_O
Y_M = l * np.sin(phi) + Y_O

X_cir = np.abs(np.sin(phi)) * l * np.cos(phi) + X_O
Y_cir = np.abs(np.sin(phi)) * l * np.sin(phi) + Y_O
#оси
X_Ground = [0, 0, 9]
Y_Ground = [9, 0, 0]

# настройки холста
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

out = FuncAnimation(fig, Kino, frames=stp, interval=10)

plt.show()
