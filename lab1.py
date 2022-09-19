import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp

def Rot2D(X, Y, Phi):
    RotX = X * np.cos(Phi) + Y * np.sin(Phi)
    RotY = X * np.sin(Phi) - Y * np.cos(Phi)
    return RotX, RotY

def vect_p(Ax, Ay, Bx, By):
    return (Ax * By - Bx * Ay)

def normalize(A):
    Ax, Ay = A
    Apif = sp.sqrt(Ax ** 2  + Ay ** 2)
    return (Ax / Apif, Ay / Apif)

def v_len(Ax, Ay):
    return sp.sqrt(Ax * Ax + Ay * Ay)

def angle(Ax, Ay, Bx, By):
    return (Ax * Bx + Ay * By) / (v_len(Ax, Ay) * v_len(Bx, By))


def turn(Ax, Ay, a):
    cos = sp.cos(a)
    sin = sp.sin(a)
    Rx = Ax * cos - Ay * sin
    Ry = Ax * sin + Ay * cos
    return (Rx, Ry)

def turn_ort(Ax, Ay):
    Rx = Ax * 0 - Ay * 1
    Ry = Ax * 1 + Ay * 0
    return (Rx, Ry)

def main():
    t = sp.Symbol('t')
    r = 2 + sp.sin(12 * t)
    phi = t + 0.2 * sp.cos(13*t)

    x = r * sp.cos(phi)
    y = r * sp.sin(phi)

    Vx = sp.diff(x, t)
    Vy = sp.diff(y, t)
    Wx = sp.diff(Vx, t)
    Wy = sp.diff(Vy, t)

    V_square = Vx * Vx + Vy * Vy
    W_square = Wx * Wx + Wy * Wy
    V_len = sp.sqrt(V_square)
    W_len = sp.sqrt(W_square)

    cos_alp = angle(Vx, Vy, Wx, Wy)
    sin_alp = sp.sqrt(1 - cos_alp * cos_alp)

    W_t = sp.diff(V_len, t)
    W_tx = (Vx / V_len) * W_t
    W_ty = (Vy / V_len) * W_t
    W_n = (W_square - W_t * W_t)
    P = (V_len * V_len * V_len) / (Vx * Wy - Wx * Vy)
    P = sp.Abs(P)


    nx, ny = normalize(turn_ort(Vx, Vy))
    directionP = vect_p(W_tx, W_ty, Wx, Wy)
    Px = nx * P
    Py = ny * P

    Func_x = sp.lambdify(t, x)
    Func_y = sp.lambdify(t, y)
    Func_Vx = sp.lambdify(t, Vx)
    Func_Vy = sp.lambdify(t, Vy)
    Func_Wx = sp.lambdify(t, Wx)
    Func_Wy = sp.lambdify(t, Wy)
    Func_Px = sp.lambdify(t, Px)
    Func_Py = sp.lambdify(t, Py)


    T = np.linspace(0, 10, 1001)
    X = np.zeros_like(T)
    Y = np.zeros_like(T)
    VX = np.zeros_like(T)
    VY = np.zeros_like(T)
    WX = np.zeros_like(T)
    WY = np.zeros_like(T)
    CX = np.zeros_like(T)
    CY = np.zeros_like(T)
    PX = np.zeros_like(T)
    PY = np.zeros_like(T)

    for i in np.arange(len(T)):
        X[i] = Func_x(T[i])
        Y[i] = Func_y(T[i])
        VX[i] = Func_Vx(T[i])
        VY[i] = Func_Vy(T[i])
        WX[i] = Func_Wx(T[i])
        WY[i] = Func_Wy(T[i])
        PX[i] = Func_Px(T[i])
        PY[i] = Func_Py(T[i])

        if i == 0:
            continue

        k = vect_p(VX[i], VY[i], VX[i - 1], VY[i - 1])
        k = -sp.sign(k)
        PX[i] *= k
        PY[i] *= k


    MaxX = np.max(X)
    MinX = np.min(X)
    MaxY = np.max(Y)
    MinY = np.min(Y)

    V_Phi = np.arctan2(VY, VX)
    W_Phi = np.arctan2(WY, WX)
    P_Phi = np.arctan2(PY, PX)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X, Y)
    ax.set(xlim=[- (MaxX - MinX) / 2 + MinX, (MaxX - MinX) / 2 + MaxX], ylim=[- (MaxY - MinY) / 2 + MinY, (MaxY - MinY) / 2 + MaxY])

    P = ax.plot(X[0], Y[0], marker='o')[0]
    V_Line = ax.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], color=[0, 0, 0])[0]
    W_Line = ax.plot([X[0], X[0] + WX[0]], [Y[0], Y[0] + WY[0]], color=[1, 1, 0])[0]
    P_Line = ax.plot([X[0], X[0] + PX[0]], [Y[0], Y[0] + PY[0]], color=[0, 0, 1])[0]

    XArrow = np.array([-0.15, 0, -0.15])
    YArrow = np.array([0.1, 0, -0.1])
    V_RArrowX, V_RArrowY = Rot2D(XArrow, YArrow, V_Phi[0])
    W_RArrowX, W_RArrowY = Rot2D(XArrow, YArrow, W_Phi[0])
    P_RArrowX, P_RArrowY = Rot2D(XArrow, YArrow, P_Phi[0])
    V_Arrow = ax.plot(X[0] + V_RArrowX, Y[0] + V_RArrowY)[0]
    W_Arrow = ax.plot(X[0] + W_RArrowX, Y[0] + W_RArrowY)[0]
    P_Arrow = ax.plot(X[0] + P_RArrowX, Y[0] + P_RArrowY)[0]

    def MovementFunction(i):
        P.set_data(X[i], Y[i])

        V_Line.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i] + VY[i]])
        W_Line.set_data([X[i], X[i] + WX[i]], [Y[i], Y[i] + WY[i]])
        P_Line.set_data([X[i], X[i] + PX[i]], [Y[i], Y[i] + PY[i]])

        V_RArrowX, V_RArrowY = Rot2D(XArrow, YArrow, V_Phi[i])
        W_RArrowX, W_RArrowY = Rot2D(XArrow, YArrow, W_Phi[i])
        P_RArrowX, P_RArrowY = Rot2D(XArrow, YArrow, P_Phi[i])
        V_Arrow.set_data(X[i] + VX[i] + V_RArrowX, Y[i] + VY[i] + V_RArrowY)
        W_Arrow.set_data(X[i] + WX[i] + W_RArrowX, Y[i] + WY[i] + W_RArrowY)
        P_Arrow.set_data(X[i] + PX[i] + P_RArrowX, Y[i] + PY[i] + P_RArrowY)
        return [P, V_Line, V_Arrow, W_Line, W_Arrow,P_Line, P_Arrow]

    output = FuncAnimation(fig, MovementFunction, interval=30, frames=len(T))

    plt.show()

if __name__ == "__main__":
    main()
