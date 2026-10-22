import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('boston_housing_real.csv', skiprows=22, sep=r'\s+', header=None)

X = df.iloc[:, 0].values
y = df.iloc[:, 9].values

print(f"Завантажено точок: {len(X)}")
print(f"Перші 2 пари: ({X[0]}, {y[0]}), ({X[1]}, {y[1]})")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Дослідна вибірка: {len(X_train)} точок")
print(f"Тестова вибірка:  {len(X_test)} точок")

x_c = np.mean(X_train)
y_c = np.mean(y_train)
print(f"Точка центру мас: X_c = {x_c:.4f}, Y_c = {y_c:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='#1f77b4', alpha=0.65, edgecolors='k', s=35, label='Спостереження (N = 506)')
plt.title('Крок 1. Множина спостережуваних значень (CRIM vs TAX)')
plt.xlabel('Стовпець 0: CRIM (рівень злочинності)')
plt.ylabel('Стовпець 9: TAX (податок на майно)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.figure(figsize=(8.5, 5.2))
plt.scatter(X_train, y_train, color='#2b5c8f', alpha=0.65, s=35, label=f'Дослідна вибірка (n = {len(X_train)})')
plt.scatter(X_test, y_test, color='#ff7f0e', alpha=0.8, edgecolors='k', s=40, marker='^', label=f'Тестова вибірка (n = {len(X_test)})')
plt.scatter([x_c], [y_c], color='#d62728', s=160, marker='*', edgecolors='black', zorder=5,
            label=f'Центр мас ({x_c:.2f}; {y_c:.2f})')
plt.axvline(x_c, color='#d62728', linestyle=':', alpha=0.6)
plt.axhline(y_c, color='#d62728', linestyle=':', alpha=0.6)
plt.title('Кроки 2–3. Розділення вибірки (80/20) та точка центру мас')
plt.xlabel('Стовпець 0: CRIM')
plt.ylabel('Стовпець 9: TAX')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')

k_vals = np.linspace(-5.0, 25.0, 61)
b_vals = np.linspace(250.0, 480.0, 47)

K_grid, B_grid = np.meshgrid(k_vals, b_vals)
MSE_grid = np.zeros_like(K_grid)
SS_grid = np.zeros_like(K_grid)

results = []

for i in range(len(b_vals)):
    for j in range(len(k_vals)):
        kv = K_grid[i, j]
        bv = B_grid[i, j]

        y_pred_tr = kv * X_train + bv

        ss_tr = np.sum((y_pred_tr - y_train) ** 2)
        mse_tr = ss_tr / len(X_train)

        SS_grid[i, j] = ss_tr
        MSE_grid[i, j] = mse_tr

        y_pred_te = kv * X_test + bv
        ss_te = np.sum((y_pred_te - y_test) ** 2)
        mse_te = ss_te / len(X_test)

        results.append({
            'k': kv, 'b': bv,
            'MSE_train': mse_tr, 'SS_train': ss_tr,
            'MSE_test': mse_te, 'SS_test': ss_te
        })

results_df = pd.DataFrame(results).sort_values(by='MSE_train').reset_index(drop=True)
best_model = results_df.iloc[0]
print(f"\nНайкраща пряма з перебору: k = {best_model['k']}, b = {best_model['b']}, MSE = {best_model['MSE_train']:.2f}")

fig = plt.figure(figsize=(10, 7))
ax3d = fig.add_subplot(111, projection='3d')

surf = ax3d.plot_surface(K_grid, B_grid, MSE_grid, cmap='viridis', alpha=0.8, edgecolor='none')
ax3d.scatter([best_model['k']], [best_model['b']], [best_model['MSE_train']],
             color='red', s=100, label=f"Мінімум MSE = {best_model['MSE_train']:.1f}")

ax3d.set_title('Крок 4d. 3D-поверхня похибки MSE(k, b)')
ax3d.set_xlabel('Коефіцієнт нахилу k')
ax3d.set_ylabel('Вільний член b')
ax3d.set_zlabel('Похибка MSE')
fig.colorbar(surf, shrink=0.5, aspect=10, label='MSE')
ax3d.legend()

print("\n=== Крок 5. Оптимальна лінійна регресія ===")
print(f"Рівняння: y = {best_model['k']:.1f} * x + {best_model['b']:.1f}")
print(f"Дослідна вибірка: MSE = {best_model['MSE_train']:.2f}, SS = {best_model['SS_train']:.2f}")

print("\n=== Крок 6. Три найближчі до оптимальної лінійні регресії ===")
nearest_3 = results_df.iloc[1:4]
print(nearest_3[['k', 'b', 'MSE_train', 'SS_train', 'MSE_test', 'SS_test']].to_string(index=False))

plt.figure(figsize=(8.5, 5.2))
plt.scatter(X_train, y_train, color='#8da0cb', alpha=0.55, s=30, label='Дослідна вибірка (n = 404)')
x_line = np.linspace(0, X_train.max(), 200)

colors_near = ['#2ca02c', '#9467bd', '#ff7f0e']
styles_near = ['--', '-.', ':']
for idx, (_, row) in enumerate(nearest_3.iterrows()):
    plt.plot(x_line, row['k'] * x_line + row['b'], color=colors_near[idx], linestyle=styles_near[idx], linewidth=1.6,
             label=f"Найближча #{idx+1}: y = {row['k']:.1f}x + {row['b']:.1f} (MSE = {row['MSE_train']:.1f})")

plt.plot(x_line, best_model['k'] * x_line + best_model['b'], color='#d62728', linewidth=2.4,
         label=f"Оптимальна (перебір): y = {best_model['k']:.1f}x + {best_model['b']:.1f} (MSE = {best_model['MSE_train']:.1f})")

plt.scatter([x_c], [y_c], color='gold', s=170, marker='*', edgecolors='black', zorder=6,
            label=f'Центр мас ({x_c:.2f}; {y_c:.2f})')

plt.title('Кроки 5–6. Оптимальна регресія та 3 найближчі прямі (дослідна вибірка)')
plt.xlabel('Стовпець 0: CRIM')
plt.ylabel('Стовпець 9: TAX')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right', fontsize=8.5)

plt.figure(figsize=(8.5, 5.2))
x_test_line = np.linspace(0, X_test.max() * 1.05, 200)

y_test_pred = best_model['k'] * X_test + best_model['b']

for xi, yi, ypi in zip(X_test, y_test, y_test_pred):
    plt.plot([xi, xi], [yi, ypi], color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

plt.scatter(X_test, y_test, color='#ff7f0e', edgecolors='k', s=45, zorder=4,
            label=f'Тестова вибірка (n = {len(X_test)})')

plt.plot(x_test_line, best_model['k'] * x_test_line + best_model['b'], color='#d62728', linewidth=2.4, zorder=5,
         label=f"Оптимальна регресія: y = {best_model['k']:.1f}x + {best_model['b']:.1f}\n(Test MSE = {best_model['MSE_test']:.1f})")

plt.title('Крок 7. Лінійна регресія на тестовій вибірці')
plt.xlabel('Стовпець 0: CRIM')
plt.ylabel('Стовпець 9: TAX')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
plt.show()