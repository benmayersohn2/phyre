from sdeint.wiener import deltaW
from scipy.optimize import root
from scipy.integrate import Radau
import numpy as np

# All methods from Tian and Burrage (2001)


def custom_fun(root_fun, u0, t_save, dW_vec=None):
    u = np.zeros((len(u0), len(t_save)))
    u[:, 0] = u0

    h = t_save[1] - t_save[0]

    if dW_vec is None:
        dW_vec = np.sqrt(h) * deltaW(len(t_save), 1, 1)

    u_old = np.array(u0)

    for i in range(1, len(t_save)):
        u_new = root(root_fun, u_old, args=(u_old, t_save[i], h, dW_vec[i]), method='hybr').x
        u[:, i] = u_new
        u_old = np.array(u_new)

    return u


def d_rk4(f, u0, t_save):

    dt = t_save[1] - t_save[0]

    u = np.zeros((len(u0), len(t_save)))
    u[:, 0] = u0

    # Butcher Tableau
    # 0     |   0       0       0       0
    # 1/2   |   1/2     0       0       0
    # 1/2   |   0       1/2     0       0
    # 1     |   0       0       1       0
    # ------------------------------------
    #       |  1/6     1/3     1/3     1/6

    c2 = c3 = 1./2
    c4 = 1.0
    a21 = 1./2
    a32 = 1./2
    a43 = 1
    b1 = b4 = 1./6
    b2 = b3 = 1./3

    for i in range(1, len(t_save)):
        f1 = f(u[:, i-1], t_save[i])
        f2 = f(u[:, i-1] + dt * a21 * f1, t_save[i] + c2 * dt)
        f3 = f(u[:, i-1] + dt * a32 * f2, t_save[i] + c3 * dt)
        f4 = f(u[:, i-1] + dt * a43 * f3, t_save[i] + c4 * dt)

        u[:, i] = u[:, i-1] + dt * (b1 * f1 + b2 * f2 + b3 * f3 + b4 * f4)

    return u


def s_implicitKP(f, g, sd, u0, t_save, dW_vec = None):
    def root_fun(x, xprev, t, dt, dW):
        return x - xprev - dt * (f(x, t) + sd * g(x, t)) - dW * g(x, t)
    return custom_fun(root_fun, u0, t_save, dW_vec=dW_vec)


def s_implicitMilsteinTaylor(f, g, sd, u0, t_save, dW_vec = None):

    def root_fun(x, xprev, t, dt, dW):
        return x - xprev - dt * f(x, t) - dW * g(x, t) - sd/2 * (dW**2 + dt) * g(x, t)

    return custom_fun(root_fun, u0, t_save, dW_vec=dW_vec)


def s_implicitModMilstein1(f, g, _, u0, t_save, dW_vec = None):

    def root_fun(x, xprev, t, dt, dW):
        return (x - xprev - dt * f(x, t) - dW * g(x, t)
                + 1./(2 * np.sqrt(dt)) * (dW**2 + 1) * (g(x + np.sqrt(dt) * g(x, t), t) - g(x, t)))

    return custom_fun(root_fun, u0, t_save, dW_vec=dW_vec)


def s_implicitModMilstein2(f, g, _, u0, t_save, dW_vec = None):

    def root_fun(x, xprev, t, dt, dW):
        return (x - xprev - dt * f(x, t) - dW * g(x, t)
                + 1./(2 * dt) * (dW**2 + 1) * (g(x + dt * g(x, t), t) - g(x, t)))

    return custom_fun(root_fun, u0, t_save, dW_vec=dW_vec)
