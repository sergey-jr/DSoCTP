from random import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import *


def plot(y, ym, yp, ypm, k):
    dpi = 200
    fig = plt.figure(dpi=dpi, figsize=(1024 / dpi, 768 / dpi))
    mpl.rcParams.update({'font.size': 10})
    plt.axis([0, N + d + 1, min(y) - 1, max(y) + 1])
    plt.title('')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.plot(x, y, color='blue', linewidth=1,
             label='y(t)')
    plt.plot(x, ym, color='red', linewidth=1,
             label='ym(t)')
    plt.legend(loc='upper right')
    fig.savefig('bezpomex{}.png'.format(k))
    fig.clear()
    plt.title('')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, yp, color='blue', linewidth=1,
             label='y(t)')
    plt.plot(x, ypm, color='red', linewidth=1,
             label='ym(t)')
    plt.legend(loc='upper right')
    fig.savefig('spomex{}.png'.format(k))


def KritFisher(y, Y, d, l):
    N = len(y)
    sr, sy, so = [0, 0, 0]
    for i in range(3 + d, N):
        sr += y[i]
    sr /= N - 2 - d
    for i in range(3 + d, N):
        sy += (y[i] - sr) ** 2
        so += (y[i] - Y[i]) ** 2
    sy /= N - 2 - d
    so /= N - 1 - d - l
    return sy / so


def form_T(N, T):
    if N == 1:
        return [T[1]]
    if N == 2:
        return [T[1] + T[2], T[1] * T[2]]
    if N == 3:
        return [T[1] + T[2] + T[3], T[1] * T[2] + T[2] * T[3] + T[1] * T[3], T[1] * T[2] * T[3]]


def form_koeff(N, T, k):
    Tk = form_T(N, T)
    if N == 1:
        return [T[0] * k / Tk[0], 1 - T[0] / Tk[0]]
    if N == 2:
        v = Tk[1] + Tk[0] * T[0]
        return [(Tk[0] * T[0] + 2 * Tk[1] - T[0] * T[0]) / v, -Tk[1] / v, k * T[0] * T[0] / v]
    if N == 3:
        v = Tk[1] * T[0] * T[0] + Tk[1] * T[0] + Tk[2]
        return [(Tk[0] * T[0] * T[0] - T[0] ** 3 + 2 * Tk[1] * T[0] + 3 * T[2]) / v, (-3 * Tk[2] + T[0] * Tk[1]) / v,
                Tk[2] / v, (k * T[0] ** 3) / v]


def form_experemental(n, d, N, a, p):
    """
    :param n: объем выборки
    :param d: целое число такта запаздывания
    :param N: порядок уровнения
    :param T: массив коэфициентов
    :return:
    """
    u = [0]
    u.extend([1 for i in range(n + d)])
    u = array(u)
    y = zeros(n + d + 1)
    yp = []
    for i in range(3 + d, n + d + 1):
        for j in range(N - 1):
            y[i] += a[j] * y[i - j - 1]
        y[i] += a[N - 1] * u[i - d - 1]
    for i in range(n + d + 1):
        m = 2 * randint(0, 1001) / 1000 - 1
        yp.append(y[i] + p * m)
    return [u, y, yp]


def form_matr(y, u, N, d):
    A = zeros((N, N))
    B = zeros(N)
    n = len(y)
    for i in range(N):
        for j in range(N):
            if j != N - 1:
                for k in range(2 + d, n):
                    A[i][j] += y[k - j - 1] * y[k - i - 1] if i != N - 1 else y[k - j - 1] * u[k - d - 1]
            else:
                for k in range(2 + d, n):
                    A[i][j] += u[k - d - 1] * y[k - i - 1] if i != N - 1 else u[k - d - 1] * u[k - d - 1]
        for k in range(2 + d, n):
            B[i] += y[k] * y[k - i - 1] if i != N - 1 else y[k] * u[k - d - 1]
    return [A, B]


def gauss(A, B, N):
    a = zeros((N, N + 1))
    for i in range(N):
        a[i] = append(A[i], B[i])

    for k in range(1, N):
        for j in range(k, N):
            m = a[j][k - 1] / a[k - 1][k - 1]
            for i in range(N + 1):
                a[j][i] -= m * a[k - 1][i]
    i = N - 1
    x = zeros(N)
    while i >= 0:
        j = N - 1
        x[i] = a[i][N] / a[i][i]
        while j > i:
            x[i] -= x[j] * a[i][j] / a[i][i]
            j -= 1
        i -= 1
    return x


f_tab = 1.2665
f = lambda x: ' '.join(['%.4f' % i for i in x])
for j in range(1, 3):
    N, n = 1000, j + 1
    t = 27
    p = 0.25
    k = -5
    T = [1, 81, 27]
    d = int(ceil(t / T[0])) + 1
    a = form_koeff(2, T, k)
    u, y, yp = form_experemental(N, d, 3, a, p)
    A, B = form_matr(y, u, n, d)
    x = gauss(A, B, n)
    print('коэффициенты модели без помех:')
    print(f(x))
    u, ym, ypm = form_experemental(N, d, n, x, p)
    x = array([i for i in range(N + d + 1)])
    f1 = KritFisher(y, ym, d, n)
    A, B = form_matr(yp, u, n, d)
    x = gauss(A, B, n)
    print('коэффициенты модели с помехами:')
    print(f(x))
    u, ym1, ypm = form_experemental(N, d, n, x, p)

    x = array([i for i in range(N + d + 1)])
    plot(y, ym, yp, ym1, j)

    f2 = KritFisher(yp, ym1, d, n)
    print('коэффициент Фишера:')
    print("%.6f(без помех), %.6f(с помехами)" % (f1, f2))

    if f1 > f_tab or f2 > f_tab:
        print('модель адекватна')
    else:
        print('модель не адекватна')
    print()
