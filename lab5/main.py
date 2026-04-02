import numpy as np
import matplotlib.pyplot as plt
from math import erf


def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)


def simpson(N, a=0, b=24):
    h = (b - a) / N
    x = a + np.arange(N + 1) * h
    y = f(x)
    return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))


def adaptive_simpson(a, b, delta, counter):
    mid = (a + b) / 2
    h = b - a
    I_full = h / 6 * (f(a) + 4 * f(mid) + f(b))
    counter[0] += 3

    def _recurse(a, b, I_ab, delta):
        mid = (a + b) / 2
        q1 = (a + mid) / 2
        q3 = (mid + b) / 2
        I_left = (mid - a) / 6 * (f(a) + 4 * f(q1) + f(mid))
        I_right = (b - mid) / 6 * (f(mid) + 4 * f(q3) + f(b))
        counter[0] += 4

        if abs(I_ab - I_left - I_right) <= delta:
            return I_left + I_right
        return _recurse(a, mid, I_left, delta / 2) + _recurse(mid, b, I_right, delta / 2)

    return _recurse(a, b, I_full, delta)


if __name__ == "__main__":
    a, b = 0, 24

    I0 = 1200 + 5 * np.sqrt(5 * np.pi) * erf(12 / np.sqrt(5))
    print("f(x) = 50 + 20*sin(pi*x/12) + 5*exp(-0.2*(x-12)^2)")
    print(f"Інтервал: [{a}, {b}]")
    print(f"Точне значення інтегралу I0 = {I0:.12f}")

    N_values = list(range(10, 1002, 2))
    errors = []
    for N in N_values:
        err = abs(simpson(N) - I0)
        errors.append(err)

    N_opt = None
    epsopt = None
    for i, (N, err) in enumerate(zip(N_values, errors)):
        if err <= 1e-12:
            N_opt = N
            epsopt = err
            break

    if N_opt is None:
        N_opt = N_values[np.argmin(errors)]
        epsopt = min(errors)

    print(f"\n{'N':>6}  {'eps(N)':>14}")
    print("-" * 24)
    for N, err in zip(N_values, errors):
        if N <= 50 or N % 100 == 0 or N == N_opt:
            print(f"  {N:4d}  {err:14.4e}")

    print(f"\nN_opt = {N_opt}  (eps <= 1e-12)")
    print(f"epsopt = |I(N_opt) - I0| = {epsopt:.4e}")

    N0_raw = N_opt / 10
    N0 = int(round(N0_raw / 8) * 8)
    if N0 < 8:
        N0 = 8
    eps0 = abs(simpson(N0) - I0)
    print(f"\nN0 = {N0}  (N_opt/10, кратне 8)")
    print(f"eps0 = |I(N0) - I0| = {eps0:.4e}")

    I_N0 = simpson(N0)
    I_N0_half = simpson(N0 // 2)
    I_R = I_N0 + (I_N0 - I_N0_half) / 15
    epsR = abs(I_R - I0)
    print("\nМетод Рунге-Ромберга:")
    print(f"I(N0)    = {I_N0:.12f}")
    print(f"I(N0/2)  = {I_N0_half:.12f}")
    print(f"I_R      = {I_R:.12f}")
    print(f"epsR     = {epsR:.4e}")
    if epsR > 0:
        print(f"Покращення: eps0/epsR ~= {eps0 / epsR:.1f} разів")

    I1 = simpson(N0)
    I2 = simpson(N0 // 2)
    I3 = simpson(N0 // 4)
    I_E = (I2**2 - I1 * I3) / (2 * I2 - (I1 + I3))
    p_aitken = (1 / np.log(2)) * np.log(abs((I3 - I2) / (I2 - I1)))
    epsE = abs(I_E - I0)
    print("\nМетод Ейткена:")
    print(f"I(N0)    = {I1:.12f}")
    print(f"I(N0/2)  = {I2:.12f}")
    print(f"I(N0/4)  = {I3:.12f}")
    print(f"I_E      = {I_E:.12f}")
    print(f"Порядок точності p ~= {p_aitken:.2f}")
    print(f"epsE     = {epsE:.4e}")
    if epsE > 0:
        print(f"Покращення: eps0/epsE ~= {eps0 / epsE:.1f} разів")

    print(f"\n{'Метод':<20} {'Похибка':>14}")
    print("-" * 36)
    print(f"{'Сімпсон (N0)':<20} {eps0:>14.4e}")
    print(f"{'Рунге-Ромберга':<20} {epsR:>14.4e}")
    print(f"{'Ейткена':<20} {epsE:>14.4e}")

    print("\nАдаптивний алгоритм Сімпсона:")
    deltas = [10**(-k) for k in range(2, 13)]
    adapt_results = []
    print(f"  {'delta':>10}  {'I_adapt':>16}  {'Похибка':>14}  {'Обчислень f':>12}")
    print("  " + "-" * 58)
    for delta in deltas:
        counter = [0]
        I_adapt = adaptive_simpson(a, b, delta, counter)
        err_adapt = abs(I_adapt - I0)
        adapt_results.append((delta, I_adapt, err_adapt, counter[0]))
        print(f"  {delta:>10.0e}  {I_adapt:>16.10f}  {err_adapt:>14.4e}  {counter[0]:>12d}")

    x_plot = np.linspace(a, b, 1000)
    plt.figure(figsize=(10, 5))
    plt.plot(x_plot, f(x_plot), "b-", lw=2,
             label=r"$f(x)=50+20\sin\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$")
    plt.xlabel("Час, x (год)")
    plt.ylabel("Навантаження, f(x)")
    plt.title("Графік функції навантаження на сервер")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lab5/fig_function.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    errors_pos = [(N, e) for N, e in zip(N_values, errors) if e > 0]
    plt.figure(figsize=(10, 5))
    plt.semilogy([x[0] for x in errors_pos], [x[1] for x in errors_pos],
                 "b-", lw=1.5, label="eps(N) = |I(N) - I0|")
    plt.axhline(y=1e-12, color="r", ls="--", lw=1, label="eps = 1e-12")
    if N_opt:
        plt.plot(N_opt, epsopt, "r*", markersize=12, label=f"N_opt = {N_opt}")
    plt.xlabel("N (число розбиттів)")
    plt.ylabel("Похибка")
    plt.title("Залежність похибки складової формули Сімпсона від N")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("lab5/fig_accuracy_vs_N.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    d_vals = [r[0] for r in adapt_results]
    e_vals = [r[2] for r in adapt_results if r[2] > 0]
    d_pos = [r[0] for r in adapt_results if r[2] > 0]
    c_vals = [r[3] for r in adapt_results]

    ax1.loglog(d_pos, e_vals, "b-o", markersize=5, lw=1.5)
    ax1.set_xlabel("delta")
    ax1.set_ylabel("Похибка")
    ax1.set_title("Точність адаптивного алгоритму")
    ax1.grid(True, alpha=0.3, which="both")

    ax2.semilogx(d_vals, c_vals, "r-s", markersize=5, lw=1.5)
    ax2.set_xlabel("delta")
    ax2.set_ylabel("Число обчислень f(x)")
    ax2.set_title("Кількість обчислень підінтегральної функції")
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("lab5/fig_adaptive.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    print("\nГрафіки: lab5/fig_*.png")
