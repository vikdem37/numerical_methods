import requests
import numpy as np
import matplotlib.pyplot as plt
# -------------------------------
# 1. Запит до Open-Elevation API
# -------------------------------
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48." \
"164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.1" \
"66777,24.531927|48.167326,24.530884|48.167011,24.530061|48.16" \
"6053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166" \
"128,24.520214|48.165416,24.517170|48.164546,24.514640|48.1634" \
"12,24.512980|48.162331,24.511715|48.162015,24.509462|48.16214" \
"7,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580" \
",24.500537|48.160250,24.500106"
response = requests.get(url)
data = response.json()

results = data["results"]
n = len(results)
print("Кількість вузлів:", n)
print("\nТабуляція вузлів:")
print("№ | Latitude | Longitude | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | "
    f"{point['longitude']:.6f} | "
    f"{point['elevation']:.2f}")


"""Обчислення кумулятивної відстані (Кумулятивна відстань (або
відстань накопичення) — це загальна сума попередніх значень, що додаються до
поточного, яка використовується для аналізу тенденцій,
накопичувальних ефектів у часових рядах, статистиці
(кумулятивна функція розподілу), або в географічних інформаційних системах 
(картографування, маршрутизація).)"""

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]
distances = [0]
for i in range(1, n):
    d = haversine(*coords[i-1], *coords[i])
    distances.append(distances[-1] + d)
print("\nТабуляція (відстань, висота):")
print("№ | Distance (m) | Elevation (m)")
with open("lab1/tabulation.txt", "w", encoding="utf-8") as f: # mk1
    f.write("Табуляція вузлів (Заросляк → Говерла)\n")
    f.write("№ | Latitude | Longitude | Elevation (m)\n")
    for i, point in enumerate(results):
        line = f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}\n"
        f.write(line)
    f.write("\nТабуляція (відстань, висота):\n")
    f.write("№ | Distance (m) | Elevation (m)\n")
    for i in range(n):
        f.write(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}\n") # mk1
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")
# cont
# Convert to numpy arrays for further use
distances = np.array(distances)
elevations = np.array(elevations)

plt.figure(figsize=(10, 5))
plt.plot(distances, elevations, "bo-", markersize=6, label="Дискретні точки GPS")
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Висота (м)")
plt.title("Профіль маршруту: Заросляк → Говерла (дискретні точки)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("lab1/fig_step5_discrete_profile.png", dpi=150)
if plt.get_backend() != "agg":
    plt.show()
plt.close()

def build_spline_tridiagonal_system(x, y):
    """
    for i=1..n-1: h_{i-1}*c_{i-1} + 2*(h_{i-1}+h_i)*c_i + h_i*c_{i+1} = rhs_i.
    Natural spline: c_0 = 0, c_n = 0.
    return lower, diag, upper (length n-1 for unknowns c_1..c_{n-1}), rhs (length n-1)
    """
    n = len(x) - 1  # number of segments (n segments, n+1 nodes)
    h = np.diff(x)
    lower = np.zeros(n - 1)
    diag = np.zeros(n - 1)
    upper = np.zeros(n - 1)
    rhs = np.zeros(n - 1)
    for i in range(1, n):
        idx = i - 1
        lower[idx] = h[i - 1]
        diag[idx] = 2.0 * (h[i - 1] + h[i])
        upper[idx] = h[i]
        rhs[idx] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    return lower, diag, upper, rhs, h


def sweep_tridiagonal(lower, diag, upper, rhs):
    """
    Thomas algorithm
    lower[i]*x[i-1] + diag[i]*x[i] + upper[i]*x[i+1] = rhs[i]
    return solution x (length = len(rhs))
    """
    n = len(rhs)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    # Forward pass
    alpha[0] = -upper[0] / diag[0]
    beta[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] + lower[i] * alpha[i - 1]
        alpha[i] = -upper[i] / denom
        beta[i] = (rhs[i] - lower[i] * beta[i - 1]) / denom
    # Backward pass
    x = np.zeros(n)
    x[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]
    return x


def natural_cubic_spline_coefficients(x, y):
    n_seg = len(x) - 1
    h = np.diff(x)
    # Build and solve for c_1..c_{n-1}; c_0 = c_n = 0
    lower, diag, upper, rhs, _ = build_spline_tridiagonal_system(x, y)
    c_inner = sweep_tridiagonal(lower, diag, upper, rhs)
    c = np.zeros(n_seg + 1)
    c[0] = 0.0
    c[1:n_seg] = c_inner
    c[n_seg] = 0.0
    # a_i = y_i
    a = y[:-1].copy()
    # d_i = (c_{i+1} - c_i) / (3*h_i)
    d = (c[1:] - c[:-1]) / (3.0 * h)
    # b_i = (y_{i+1}-y_i)/h_i - c_i*h_i - d_i*h_i^2
    b = (y[1:] - y[:-1]) / h - c[:-1] * h - d * (h ** 2)
    return a, b, c[:-1], d  # c returned for segment i is c_i (length n_seg)


def spline_evaluate(x_nodes, a, b, c, d, x_query):
    x_nodes = np.asarray(x_nodes)
    a, b, c, d = np.asarray(a), np.asarray(b), np.asarray(c), np.asarray(d)
    y_out = np.zeros_like(x_query, dtype=float)
    for k, xv in enumerate(x_query):
        i = np.searchsorted(x_nodes[1:], xv)
        if i >= len(x_nodes) - 1:
            i = len(x_nodes) - 2
        t = xv - x_nodes[i]
        y_out[k] = a[i] + b[i] * t + c[i] * (t ** 2) + d[i] * (t ** 3)
    return y_out


def spline_derivative(x_nodes, a, b, c, d, x_query):
    x_nodes = np.asarray(x_nodes)
    y_out = np.zeros_like(x_query, dtype=float)
    for k, xv in enumerate(x_query):
        i = np.searchsorted(x_nodes[1:], xv)
        if i >= len(x_nodes) - 1:
            i = len(x_nodes) - 2
        t = xv - x_nodes[i]
        y_out[k] = b[i] + 2 * c[i] * t + 3 * d[i] * (t ** 2)
    return y_out


# Build tridiagonal system for full data (for console output)
lower, diag, upper, rhs, h_full = build_spline_tridiagonal_system(distances, elevations)
print("\nКоефіцієнти СЛАР для {c_i} (трьохдіагональна матриця)")
print("Нижня діагональ (lower):", lower)
print("Головна діагональ (diag):", diag)
print("Верхня діагональ (upper):", upper)
print("Вільні члени (rhs):", rhs)

# Step 7: Solve for c_i and output
c_inner = sweep_tridiagonal(lower, diag, upper, rhs)
n_seg = len(distances) - 1
c_full = np.zeros(n_seg + 1)
c_full[0] = 0.0
c_full[1:n_seg] = c_inner
c_full[n_seg] = 0.0
print("\nРозв'язок методом прогонки — значення c_i")
for i in range(len(c_full)):
    print(f"  c_{i} = {c_full[i]:.6g}")

# Steps 8–9: Compute a, b, c, d and output
a_full, b_full, c_seg, d_full = natural_cubic_spline_coefficients(distances, elevations)
print("\nКоефіцієнти кубічних сплайнів a_i, b_i, c_i, d_i")
print("  i  |    a_i     |    b_i     |    c_i     |    d_i     ")
print("-----+------------+------------+------------+------------")
for i in range(len(a_full)):
    print(f" {i:2d}  | {a_full[i]:10.4f} | {b_full[i]:10.4f} | {c_seg[i]:10.6f} | {d_full[i]:10.6e}")

def subsample_nodes(distances_all, elevations_all, num_nodes):
    n_all = len(distances_all)
    if num_nodes >= n_all:
        return distances_all, elevations_all
    indices = np.linspace(0, n_all - 1, num_nodes, dtype=int)
    indices = np.unique(indices)
    return distances_all[indices], elevations_all[indices]

xx_fine = np.linspace(distances[0], distances[-1], 500)

for num_nodes in [10, 15, 20]:
    x_sub, y_sub = subsample_nodes(distances, elevations, num_nodes)
    a_n, b_n, c_n, d_n = natural_cubic_spline_coefficients(x_sub, y_sub)
    yy_interp = spline_evaluate(x_sub, a_n, b_n, c_n, d_n, xx_fine)
    # Reference: full data as "given function"
    yy_ref = spline_evaluate(distances, a_full, b_full, c_seg, d_full, xx_fine)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(distances, elevations, "k.", markersize=4, label="Усі вузли")
    axes[0].plot(x_sub, y_sub, "ro", markersize=8, label=f"Вибрані вузли (n={len(x_sub)})")
    axes[0].plot(xx_fine, yy_interp, "b-", lw=2, label="Сплайн (інтерполяція)")
    axes[0].set_xlabel("Кумулятивна відстань (м)")
    axes[0].set_ylabel("Висота (м)")
    axes[0].set_title(f"Профіль маршруту: {num_nodes} вузлів")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    err = yy_interp - yy_ref
    axes[1].plot(xx_fine, err, "r-", lw=1)
    axes[1].axhline(0, color="k", ls="--", alpha=0.5)
    axes[1].set_xlabel("Кумулятивна відстань (м)")
    axes[1].set_ylabel("Похибка (м)")
    axes[1].set_title(f"Похибка інтерполяції (від повного сплайна), n={num_nodes}")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"lab1/fig_step10_{num_nodes}_nodes.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

yy_ref_full = spline_evaluate(distances, a_full, b_full, c_seg, d_full, xx_fine)
print("\nВплив кількості вузлів на точність")
for num_nodes in [10, 15, 20]:
    x_sub, y_sub = subsample_nodes(distances, elevations, num_nodes)
    a_n, b_n, c_n, d_n = natural_cubic_spline_coefficients(x_sub, y_sub)
    yy_interp = spline_evaluate(x_sub, a_n, b_n, c_n, d_n, xx_fine)
    err = yy_interp - yy_ref_full
    print(f"  n = {num_nodes}: max|похибка| = {np.max(np.abs(err)):.4f} м, середня |похибка| = {np.mean(np.abs(err)):.4f} м")

x_15, y_15 = subsample_nodes(distances, elevations, 15)
a_15, b_15, c_15, d_15 = natural_cubic_spline_coefficients(x_15, y_15)
y_given = spline_evaluate(distances, a_full, b_full, c_seg, d_full, xx_fine)
y_approx = spline_evaluate(x_15, a_15, b_15, c_15, d_15, xx_fine)
error_12 = y_approx - y_given

fig12, axes12 = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
axes12[0].plot(xx_fine, y_given, "b-", lw=2, label="Задана функція y(x) (сплайн по всіх вузлах)")
axes12[0].set_ylabel("Висота (м)")
axes12[0].set_title("Задана функція y(x)")
axes12[0].legend()
axes12[0].grid(True, alpha=0.3)

axes12[1].plot(xx_fine, y_approx, "g-", lw=2, label="Наближення (15 вузлів)")
axes12[1].set_ylabel("Висота (м)")
axes12[1].set_title("Наближені значення")
axes12[1].legend()
axes12[1].grid(True, alpha=0.3)

axes12[2].plot(xx_fine, error_12, "r-", lw=1, label="Похибка")
axes12[2].axhline(0, color="k", ls="--", alpha=0.5)
axes12[2].set_xlabel("Кумулятивна відстань (м)")
axes12[2].set_ylabel("Похибка (м)")
axes12[2].set_title("Похибка на відрізку")
axes12[2].legend()
axes12[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lab1/fig_step12_y_approx_error.png", dpi=150)
if plt.get_backend() != "agg":
    plt.show()
plt.close()

total_distance = distances[-1] - distances[0]
total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, len(elevations)))
total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, len(elevations)))
print("\nХарактеристики маршруту")
print(f"  Загальна довжина маршруту (м): {total_distance:.2f}")
print(f"  Сумарний набір висоти (м): {total_ascent:.2f}")
print(f"  Сумарний спуск (м): {total_descent:.2f}")

# Gradient via spline derivative (elevation change per distance -> %)
xx_grad = np.linspace(distances[0], distances[-1], 1000)
dy_dx = spline_derivative(distances, a_full, b_full, c_seg, d_full, xx_grad)
grad_percent = dy_dx * 100  # rise/run: (m elevation / m distance) * 100 = %
print("\nАналіз градієнта (похідна сплайна)")
print(f"  Максимальний підйом (%): {np.max(grad_percent):.2f}")
print(f"  Максимальний спуск (%): {np.min(grad_percent):.2f}")
print(f"  Середній градієнт (%): {np.mean(np.abs(grad_percent)):.2f}")
steep = np.where(np.abs(grad_percent) > 15)[0]
if len(steep) > 0:
    dist_steep = xx_grad[steep]
    print(f"  Ділянки з крутизною > 15%: відстані (м) {dist_steep[0]:.1f} — {dist_steep[-1]:.1f}")
else:
    print("  Ділянок з крутизною > 15% немає.")

# Mechanical energy (80 kg)
mass_kg = 80
g = 9.81
energy_j = mass_kg * g * total_ascent
print("\nМеханічна енергія підйому (маса 80 кг)")
print(f"  Механічна робота (Дж): {energy_j:.2f}")
print(f"  Механічна робота (кДж): {energy_j/1000:.2f}")
print(f"  Енергія (ккал): {energy_j/4184:.2f}")

print("\nГрафіки збережено в lab1/ (fig_step5_*, fig_step10_*, fig_step12_*).")
