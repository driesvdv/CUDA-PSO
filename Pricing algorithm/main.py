import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PSO import PSO

def calculate_price(offset_1, offset_2):
    prices = np.zeros(len(offset_1), dtype=float)

    for i in range(len(offset_1)):
        dishw_price = np.sum(dishwasher * pricing[offset_1[i]:offset_1[i] + len(dishwasher)])
        oven_price = np.sum(oven * pricing[offset_2[i]:offset_2[i] + len(oven)])
        prices[i] = dishw_price + oven_price
    return prices

def fitness_function(offsets):
    return calculate_price(offsets[:, 0], offsets[:, 1])

# Read ../data/dyn_pricing.csv
pricing = pd.read_csv('data/dyn_pricing_1.csv')
# Only keep the price column and save to array
pricing = pricing['price'].values / 1000

# Read ../data/dishw_1.csv
dishwasher = pd.read_csv('data/dishw_1.csv')
# Only keep the power column and save to array and convert to kW
dishwasher = dishwasher['value'].values / 1000

# Read ../data/oven_1.csv
oven = pd.read_csv('data/oven_1.csv')
# Only keep the power column and save to array and convert to EUR/kWh
oven = oven['value'].values / 1000

# Initialize PSO
n_particles = 100
particles = np.random.randint(0, 15, size=(n_particles, 2))  # Random initial offsets
velocities = np.random.uniform(-1, 1, size=(n_particles, 2))  # Random initial velocities

pso = PSO(particles, velocities, fitness_function)

# Run PSO
while pso.next():
    pass

# Get best solution
best_offsets = pso.g_best
best_price = pso.g_best_value

print("Best Offsets:", best_offsets)
print("Best Price:", best_price)

# Plot calculate price for different offsets (0..15) for device 1 and 2 in a 3D meshgrid
X = np.arange(0, 15, 1)
Y = np.arange(0, 15, 1)
meshgrid = np.meshgrid(X, Y)
# Create Z as a 2D array corresponding to the grid points
# Create Z as a 2D array corresponding to the grid points
Z = np.zeros((len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        Z[i, j] = calculate_price(np.array([X[i]]), np.array([Y[j]]))[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the surface
surf = ax.plot_surface(*meshgrid, Z, cmap='plasma', edgecolor='none', alpha=0.8)
# Add labels
ax.set_xlabel('Offset for Dishwasher')
ax.set_ylabel('Offset for Oven')
ax.set_zlabel('Price')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

 # Plot the best solution found by PSO
ax.scatter(best_offsets[1], best_offsets[0], best_price, color='red', marker='o', label='Best Solution')

plt.legend()
plt.show()
