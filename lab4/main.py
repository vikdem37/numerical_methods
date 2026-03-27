import numpy as np
import matplotlib.pyplot as plt


def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)


def M_deriv(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)


def central_diff(f, x0, h):
    return (f(x0 + h) - f(x0 - h)) / (2 * h)


def runge_romberg(Dh, Dh2, p=2):
    return Dh2 + (Dh2 - Dh) / (2**p - 1)


def aitken(Dh, Dh2, Dh4):
    D_star = Dh - (Dh2 - Dh)**2 / (Dh4 - 2 * Dh2 + Dh)
    p = (1 / np.log(2)) * np.log(abs((Dh4 - Dh2) / (Dh2 - Dh)))
    return D_star, p


if __name__ == "__main__":
    t0 = 1.0
    exact = M_deriv(t0)

    print("M(t) = 50*exp(-0.1*t) + 5*sin(t)")
    print("M'(t) = -5*exp(-0.1*t) + 5*cos(t)")
    print(f"Точне значення M'({t0:.0f}) = {exact:.10f}")

    h_values = [10**k for k in range(1, -21, -1)]
    errors = []
    print(f"\n{'h':>12}  {'D(h)':>16}  {'Похибка':>14}")
    print("-" * 46)
    for h in h_values:
        Dh = central_diff(M, t0, h)
        err = abs(Dh - exact)
        errors.append(err)
        print(f"  {h:>10.0e}  {Dh:>16.10f}  {err:>14.2e}")

    best_idx = np.argmin(errors)
    h0 = h_values[best_idx]
    R0 = errors[best_idx]
    print(f"\nОптимальний крок: h0 = {h0:.0e}")
    print(f"Найкраща точність: R0 = {R0:.2e}")

    h = 0.001
    print(f"\nПриймаємо h = {h}")

    Dh = central_diff(M, t0, h)
    D2h = central_diff(M, t0, 2 * h)
    R1 = abs(Dh - exact)
    print(f"\nD(h)  = {Dh:.10f}")
    print(f"D(2h) = {D2h:.10f}")
    print(f"Похибка R1 = |D(h) - M'(t0)| = {R1:.2e}")

    D_rr = runge_romberg(D2h, Dh, p=2)
    R2 = abs(D_rr - exact)
    print("\nМетод Рунге-Ромберга:")
    print(f"D = D(h) + (D(h) - D(2h)) / (2^2 - 1) = {D_rr:.10f}")
    print(f"Похибка R2 = {R2:.2e}")
    if R2 > 0:
        print(f"Покращення: R1/R2 ~= {R1 / R2:.1f} разів")

    D4h = central_diff(M, t0, 4 * h)
    D_aitken, p_order = aitken(Dh, D2h, D4h)
    R3 = abs(D_aitken - exact)
    print("\nМетод Ейткена:")
    print(f"D(h)  = {Dh:.10f}")
    print(f"D(2h) = {D2h:.10f}")
    print(f"D(4h) = {D4h:.10f}")
    print(f"D*    = {D_aitken:.10f}")
    print(f"Оцінка порядку точності: p ~= {p_order:.2f}")
    print(f"Похибка R3 = {R3:.2e}")
    if R3 > 0:
        print(f"Покращення: R1/R3 ~= {R1 / R3:.1f} разів")

    print(f"\nВисновок: M'({t0:.0f}) ~= {exact:.4f} < 0")

    h_plot = [h_values[i] for i in range(len(h_values)) if errors[i] > 0]
    e_plot = [errors[i] for i in range(len(h_values)) if errors[i] > 0]

    plt.figure(figsize=(10, 5))
    plt.loglog(h_plot, e_plot, "b-o", markersize=5, lw=1.5, label="Похибка |D(h) - M'(t0)|")
    plt.loglog(h0, R0, "r*", markersize=15, label=f"Оптимальне h0 = {h0:.0e}")
    plt.xlabel("Крок h")
    plt.ylabel("Похибка")
    plt.title("Залежність похибки чисельного диференціювання від кроку")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("lab4/fig_error_vs_h.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    t_range = np.linspace(0, 20, 500)
    tangent_x = np.linspace(t0 - 3, t0 + 3, 100)
    tangent_y = M(t0) + exact * (tangent_x - t0)

    plt.figure(figsize=(10, 5))
    plt.plot(t_range, M(t_range), "b-", lw=2, label="M(t)")
    plt.plot(tangent_x, tangent_y, "r--", lw=1.5, label=f"Дотична в t0={t0:.0f}")
    plt.plot(t0, M(t0), "ko", markersize=8)
    plt.xlabel("t")
    plt.ylabel("M(t)")
    plt.title("Вологість ґрунту M(t) та дотична")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lab4/fig_function.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    print("\nГрафіки: lab4/fig_*.png")
