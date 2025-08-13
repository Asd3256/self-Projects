import numpy as np
import matplotlib.pyplot as plt

# Define the range for ξ and η
xi_values = np.linspace(0, 1, 101)  # 101 steps (including 0 and 1)
eta_values = np.linspace(0, 1, 81)  # 81 steps (including 0 and 1)

# Store computed values
x_grid = []
y_grid = []

# Iterate over the grid and compute values
for eta in eta_values:
    x_row = []
    y_row = []
    for xi in xi_values:
        x = (19 * eta + 1) * np.cos(2 * np.pi * xi)
        y = ((1 + 39 * eta) / 2) * np.sin(2 * np.pi * xi)
        x_row.append(x)
        y_row.append(y)
    x_grid.append(x_row)
    y_grid.append(y_row)

# Plot the result
plt.figure(figsize=(8, 6))
for i in range(len(eta_values)):
    plt.plot(x_grid[i], y_grid[i], color='blue', linewidth=0.5)  # Radial lines
for j in range(len(xi_values)):
    plt.plot([x_grid[i][j] for i in range(len(eta_values))], [y_grid[i][j] for i in range(len(eta_values))], color='red', linewidth=0.5)  # Ellipse shape

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Connected Plot of Computed Points')
plt.axis('equal')  # Set aspect ratio to ensure last ellipse looks more circular
plt.grid()
plt.show()

# Print first 10 results as a preview
for row in list(zip(xi_values, eta_values, x_grid[0], y_grid[0]))[:10]:
    print(f"ξ: {row[0]:.3f}, η: {row[1]:.3f} -> x: {row[2]:.3f}, y: {row[3]:.3f}")
