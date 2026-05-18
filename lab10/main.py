import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return y - x**2 + 1


def y_exact(x):
    return (x + 1)**2 - 0.5 * np.exp(x)


def rk_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk_solve(x0, y0, xn, h):
    n = int(round((xn - x0) / h))
    xs = np.zeros(n + 1)
    ys = np.zeros(n + 1)
    xs[0], ys[0] = x0, y0
    for i in range(n):
        xs[i + 1] = xs[i] + h
        ys[i + 1] = rk_step(xs[i], ys[i], h)
    return xs, ys


def adams2_solve(x0, y0, xn, h):
    n = int(round((xn - x0) / h))
    xs = np.zeros(n + 1)
    ys = np.zeros(n + 1)
    y_pr = np.zeros(n + 1)
    y_cr = np.zeros(n + 1)

    xs[0], ys[0] = x0, y0
    xs[1] = x0 + h
    ys[1] = rk_step(x0, y0, h)

    fs = np.zeros(n + 1)
    fs[0] = f(xs[0], ys[0])
    fs[1] = f(xs[1], ys[1])

    for i in range(1, n):
        xs[i + 1] = xs[i] + h

        yp = ys[i] + h / 2 * (3 * fs[i] - fs[i - 1])
        y_pr[i + 1] = yp

        yc = yp
        for _ in range(20):
            yc_new = ys[i] + h / 2 * (f(xs[i + 1], yc) + fs[i])
            if abs(yc_new - yc) <= 1e-14:
                yc = yc_new
                break
            yc = yc_new
        y_cr[i + 1] = yc

        ys[i + 1] = yc - (yc - yp) / 6
        fs[i + 1] = f(xs[i + 1], ys[i + 1])

    return xs, ys, y_pr, y_cr


def adams2_adaptive(x0, y0, xn, h0, eps):
    xs = [x0]
    ys = [y0]
    hs = []

    h = h0
    x, y = x0, y0

    y1 = rk_step(x, y, h)
    x1 = x + h
    xs.append(x1)
    ys.append(y1)
    hs.append(h)

    f_prev = f(x, y)
    f_curr = f(x1, y1)

    while xs[-1] < xn - 1e-12:
        x_n = xs[-1]
        y_n = ys[-1]

        if x_n + h > xn + 1e-12:
            h = xn - x_n

        x_next = x_n + h

        yp = y_n + h / 2 * (3 * f_curr - f_prev)

        yc = yp
        for _ in range(20):
            yc_new = y_n + h / 2 * (f(x_next, yc) + f_curr)
            if abs(yc_new - yc) <= 1e-14:
                yc = yc_new
                break
            yc = yc_new

        err = abs(yc - yp) / 6

        if err > eps:
            h /= 2
            y_mid = rk_step(xs[-2], ys[-2], h)
            xs[-1] = xs[-2] + h
            ys[-1] = y_mid
            hs[-1] = h
            f_prev = f(xs[-2], ys[-2])
            f_curr = f(xs[-1], ys[-1])
            continue

        y_final = yc - (yc - yp) / 6
        xs.append(x_next)
        ys.append(y_final)
        hs.append(h)

        f_prev = f_curr
        f_curr = f(x_next, y_final)

        if err < eps / 32 and h * 2 <= h0 * 4:
            h *= 2

    return np.array(xs), np.array(ys), np.array(hs)


def rk_adaptive(x0, y0, xn, h0, eps):
    xs = [x0]
    ys = [y0]
    hs = []
    h = h0
    x, y = x0, y0

    while x < xn - 1e-12:
        if x + h > xn + 1e-12:
            h = xn - x

        y_full = rk_step(x, y, h)

        y_half = rk_step(x, y, h / 2)
        y_half = rk_step(x + h / 2, y_half, h / 2)

        err = 16 / 15 * abs(y_half - y_full)

        if err > eps:
            h /= 2
            continue

        xs.append(x + h)
        ys.append(y_half)
        hs.append(h)
        x += h
        y = y_half

        if err < eps / 32:
            h *= 2

    return np.array(xs), np.array(ys), np.array(hs)


if __name__ == "__main__":
    a, b_end = 0.0, 2.0
    y0 = 0.5

    print("Задача Кошi: y' = y - x^2 + 1, y(0) = 0.5")
    print("Точний розв'язок: y(x) = (x+1)^2 - 0.5*exp(x)")
    print(f"Iнтервал: [{a}, {b_end}]")

    h_adams = 0.01
    xs_a, ys_a, yp_a, yc_a = adams2_solve(a, y0, b_end, h_adams)
    ye_a = y_exact(xs_a)
    err_a = ys_a - ye_a

    print(f"\nh = {h_adams}")
    print(f"{'x':>6}  {'y_числ':>14}  {'y_точн':>14}  {'похибка':>14}")
    print("-" * 52)
    for i in range(0, len(xs_a), max(1, len(xs_a) // 10)):
        print(f"{xs_a[i]:6.2f}  {ys_a[i]:14.8f}  {ye_a[i]:14.8f}  {err_a[i]:14.4e}")
    print(f"max|похибка| = {np.max(np.abs(err_a)):.4e}")

    plt.figure(figsize=(10, 5))
    plt.plot(xs_a, err_a, "b-", lw=1.5, label="y_n - y(x_n)")
    plt.axhline(y=0, color="k", lw=0.5)
    plt.xlabel("x")
    plt.ylabel("Локальна похибка")
    plt.title(f"Адамс 2-го порядку: локальна похибка (h = {h_adams})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lab10/fig_adams_error.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    diff_pc = yc_a[2:] - yp_a[2:]

    plt.figure(figsize=(10, 5))
    plt.plot(xs_a[2:], np.abs(err_a[2:]), "b-", lw=1.5,
             label="|y_n - y(x_n)| (точна)")
    plt.plot(xs_a[2:], np.abs(diff_pc) / 6, "r--", lw=1.5,
             label="|y_cor - y_pr| / 6 (оцiнка)")
    plt.xlabel("x")
    plt.ylabel("Похибка")
    plt.title("Адамс: оцiнка похибки vs точна похибка")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("lab10/fig_adams_estimate.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    eps_adapt = 1e-6
    print(f"\nАдаптивний крок (eps = {eps_adapt:.0e}):")
    xs_aa, ys_aa, hs_aa = adams2_adaptive(a, y0, b_end, h_adams, eps_adapt)
    err_aa = ys_aa - y_exact(xs_aa)
    print(f"  Крокiв: {len(hs_aa)}")
    print(f"  h_min = {np.min(hs_aa):.6f}, h_max = {np.max(hs_aa):.6f}")
    print(f"  max|похибка| = {np.max(np.abs(err_aa)):.4e}")

    plt.figure(figsize=(10, 5))
    plt.plot(xs_aa[1:], hs_aa, "b-", lw=1.5)
    plt.xlabel("x")
    plt.ylabel("h")
    plt.title(f"Адамс: автоматичний вибiр кроку (eps = {eps_adapt:.0e})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lab10/fig_adams_adaptive.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    h_rk = 0.01
    xs_r, ys_r = rk_solve(a, y0, b_end, h_rk)
    ye_r = y_exact(xs_r)
    err_r = ys_r - ye_r

    print(f"\nh = {h_rk}")
    print(f"{'x':>6}  {'y_числ':>14}  {'y_точн':>14}  {'похибка':>14}")
    print("-" * 52)
    step = max(1, len(xs_r) // 10)
    for i in range(0, len(xs_r), step):
        print(f"{xs_r[i]:6.2f}  {ys_r[i]:14.8f}  {ye_r[i]:14.8f}  {err_r[i]:14.4e}")
    print(f"max|похибка| = {np.max(np.abs(err_r)):.4e}")

    plt.figure(figsize=(10, 5))
    plt.plot(xs_r, err_r, "b-", lw=1.5, label="y_n - y(x_n)")
    plt.axhline(y=0, color="k", lw=0.5)
    plt.xlabel("x")
    plt.ylabel("Локальна похибка")
    plt.title(f"Рунге-Кутта: локальна похибка (h = {h_rk})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lab10/fig_rk_error.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    print("\nЗалежнiсть похибки вiд кроку:")
    h_values = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    max_errors = []
    print(f"  {'h':>8}  {'max|err|':>14}  {'порядок':>8}")
    print("  " + "-" * 34)
    for hv in h_values:
        xs_t, ys_t = rk_solve(a, y0, b_end, hv)
        me = np.max(np.abs(ys_t - y_exact(xs_t)))
        max_errors.append(me)

    for i, (hv, me) in enumerate(zip(h_values, max_errors)):
        if i > 0 and max_errors[i - 1] > 0 and me > 0:
            order = np.log(max_errors[i - 1] / me) / np.log(h_values[i - 1] / hv)
            print(f"  {hv:8.4f}  {me:14.4e}  {order:8.2f}")
        else:
            print(f"  {hv:8.4f}  {me:14.4e}  {'---':>8}")

    plt.figure(figsize=(10, 5))
    plt.loglog(h_values, max_errors, "b-o", markersize=6, lw=1.5, label="max|y_n - y(x_n)|")
    h_arr = np.array(h_values)
    c4 = max_errors[-1] / h_values[-1]**4
    plt.loglog(h_arr, c4 * h_arr**4, "r--", lw=1, alpha=0.7, label="O(h^4)")
    plt.xlabel("h")
    plt.ylabel("max|похибка|")
    plt.title("Рунге-Кутта: залежнiсть похибки вiд кроку")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("lab10/fig_rk_convergence.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    print("\nОцiнка похибки методом Рунге:")
    xs_h, ys_h = rk_solve(a, y0, b_end, h_rk)
    xs_h2, ys_h2 = rk_solve(a, y0, b_end, h_rk / 2)
    runge_err = 16 / 15 * (ys_h2[::2] - ys_h)
    true_err = ys_h - y_exact(xs_h)

    print(f"  max|оцiнка Рунге| = {np.max(np.abs(runge_err)):.4e}")
    print(f"  max|точна похибка| = {np.max(np.abs(true_err)):.4e}")

    target_eps = 1e-8
    h_needed = h_rk * (target_eps / np.max(np.abs(runge_err)))**0.25
    print(f"\n  Для eps = {target_eps:.0e} потрiбен крок h ~= {h_needed:.6f}")

    plt.figure(figsize=(10, 5))
    plt.plot(xs_h, np.abs(true_err), "b-", lw=1.5, label="|y_n - y(x_n)| (точна)")
    plt.plot(xs_h, np.abs(runge_err), "r--", lw=1.5, label="16/15 |y^{h/2} - y^h| (Рунге)")
    plt.xlabel("x")
    plt.ylabel("Похибка")
    plt.title(f"Рунге-Кутта: оцiнка Рунге vs точна похибка (h = {h_rk})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("lab10/fig_rk_runge.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    eps_rk_adapt = 1e-6
    print(f"\nАдаптивний крок Рунге-Кутта (eps = {eps_rk_adapt:.0e}):")
    xs_ra, ys_ra, hs_ra = rk_adaptive(a, y0, b_end, 0.1, eps_rk_adapt)
    err_ra = ys_ra - y_exact(xs_ra)
    print(f"  Крокiв: {len(hs_ra)}")
    print(f"  h_min = {np.min(hs_ra):.6f}, h_max = {np.max(hs_ra):.6f}")
    print(f"  max|похибка| = {np.max(np.abs(err_ra)):.4e}")

    plt.figure(figsize=(10, 5))
    plt.plot(xs_ra[1:], hs_ra, "b-o", markersize=3, lw=1.5)
    plt.xlabel("x")
    plt.ylabel("h")
    plt.title(f"Рунге-Кутта: автоматичний вибiр кроку (eps = {eps_rk_adapt:.0e})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lab10/fig_rk_adaptive.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    print(f"\n{'Метод':<25} {'h':>8} {'max|похибка|':>14} {'Крокiв':>8}")
    print("-" * 58)
    print(f"{'Адамс 2-го пор.':<25} {h_adams:>8.4f} {np.max(np.abs(err_a)):>14.4e} {len(xs_a)-1:>8d}")
    print(f"{'Рунге-Кутта':<25} {h_rk:>8.4f} {np.max(np.abs(err_r)):>14.4e} {len(xs_r)-1:>8d}")
    print(f"{'Адамс (адапт.)':<25} {'---':>8} {np.max(np.abs(err_aa)):>14.4e} {len(hs_aa):>8d}")
    print(f"{'Рунге-Кутта (адапт.)':<25} {'---':>8} {np.max(np.abs(err_ra)):>14.4e} {len(hs_ra):>8d}")

    print("\nГрафiки: lab10/fig_*.png")
