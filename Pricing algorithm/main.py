import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PSO import PSO

def calculate_price(particles):
    ev_consumption = particles[:, :24]
    dishwasher_offset = particles[:, 24]

    prices = np.zeros(n_particles, dtype=float)

    # For loop to be parallelized in cuda
    for i in range(n_particles):
        max_power_usage_per_hour = np.zeros(24, dtype=float)

        ev_price = 999999
        dishw_price = 999999

        if dishwasher_offset[i] >= 0 and dishwasher_offset[i] < len(pricing) - 9:
            dishw_price = np.sum(dishwasher * pricing[dishwasher_offset[i]:dishwasher_offset[i] + len(dishwasher)])
            max_power_usage_per_hour[dishwasher_offset[i]:dishwasher_offset[i] + len(dishwasher)] += dishwasher

        tmp = np.sum(ev_consumption[i])
        if ev_consumption[i].min() >= 0 and tmp == 30:
            ev_price = np.sum(ev_consumption[i] * pricing)
            max_power_usage_per_hour += ev_consumption[i]

        # Enforce max energy consumption of 5 at each ev_consumption[i]
        if max_power_usage_per_hour.max() > 10:
            prices[i] = 999999
        else:
            prices[i] = ev_price + dishw_price

    return prices

def fitness_function(offsets):
    return calculate_price(offsets[:, :])
    #return calculate_price(offsets[:, 0], offsets[:, 1], offsets[:, 2])

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

# Read ../data/cooktop_1.csv
cooktop = pd.read_csv('data/cooktop_1.csv')
# Only keep the power column and save to array and convert to EUR/kWh
cooktop = cooktop['value'].values / 1000

# Add an EV as device (resolution is in kWh)
ev = np.zeros(len(pricing), dtype=int)


# Initialize PSO
n_particles = 2000
particles = np.random.randint(0, 10, size=(n_particles, 25))  # Random initial offsets
velocities = np.random.uniform(-1, 10, size=(n_particles, 25))  # Random initial velocities

pso = PSO(particles, velocities, fitness_function)

# Run PSO
while pso.next():
    pass

# Get best solution
best_particles = pso.g_best
best_price = pso.g_best_value

# Convert best particles to usage array for grid
ev_consumption = best_particles[:24]
dishwasher_offset = best_particles[24]

total_power_usage = np.zeros(24, dtype=float)
total_power_usage += ev_consumption
total_power_usage[dishwasher_offset:dishwasher_offset + len(dishwasher)] += dishwasher

# Plot the total_power_usage
plt.bar(range(24), total_power_usage, label=f'Best Price: {best_price}')
plt.xlabel('Hour')
plt.ylabel('Power Usage (kW)')
plt.title('Total Power Usage')

# Add legend
plt.legend()

plt.show()


print("\nBest Price:", best_price)

# # Plot calculate price for different offsets (0..15) for device 1 and 2 in a 3D meshgrid
# X = np.arange(0, 15, 1)
# Y = np.arange(0, 15, 1)
# meshgrid = np.meshgrid(X, Y)
# # Create Z as a 2D array corresponding to the grid points
# # Create Z as a 2D array corresponding to the grid points
# Z = np.zeros((len(X), len(Y)))
# for i in range(len(X)):
#     for j in range(len(Y)):
#         Z[i, j] = calculate_price(np.array([X[i]]), np.array([Y[j]]))[0]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Plot the surface
# surf = ax.plot_surface(*meshgrid, Z, cmap='plasma', edgecolor='none', alpha=0.8)
# # Add labels
# ax.set_xlabel('Offset for Dishwasher')
# ax.set_ylabel('Offset for Oven')
# ax.set_zlabel('Price')
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
#  # Plot the best solution found by PSO
# ax.scatter(best_offsets[1], best_offsets[0], best_price, color='red', marker='o', label='Best Solution')
# plt.legend()
# plt.show()
