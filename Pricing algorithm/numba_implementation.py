import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from numba import cuda, float32

import sys
import numpy as np

@cuda.jit
def calculate_price_kernel(pricing, dishwasher, particles, prices):
    i = cuda.grid(1)
    if i < particles.shape[0]:  # Check array boundaries
        ev_consumption = particles[i, :24]
        dishwasher_offset = particles[i, 24]

        max_power_usage_per_hour = cuda.local.array(24, dtype=float32)

        ev_price = 999999
        dishw_price = 999999

        if dishwasher_offset >= 0 and dishwasher_offset < len(pricing) - 9:
            dishw_price = np.sum(dishwasher * pricing[dishwasher_offset:dishwasher_offset + len(dishwasher)])
            max_power_usage_per_hour[dishwasher_offset:dishwasher_offset + len(dishwasher)] += dishwasher

        tmp = np.sum(ev_consumption)
        if ev_consumption.min() >= 0 and tmp == 30:
            ev_price = np.sum(ev_consumption * pricing)
            max_power_usage_per_hour += ev_consumption

        # Enforce max energy consumption of 5 at each ev_consumption[i]
        if max_power_usage_per_hour.max() > 5:
            prices[i] = 999999
        else:
            prices[i] = ev_price + dishw_price

class PSO:

    def __init__(self, particles, velocities, fitness_function,
                 w=0.8, c_1=1, c_2=1, max_iter=100, auto_coef=True):
        self.particles = particles
        self.velocities = velocities
        self.fitness_function = fitness_function

        self.N = len(self.particles)
        self.w = w
        self.c_1 = c_1
        self.c_2 = c_2
        self.auto_coef = auto_coef
        self.max_iter = max_iter


        self.p_bests = self.particles
        self.p_bests_values = self.fitness_function(self.particles)
        self.g_best = self.p_bests[0]
        self.g_best_value = self.p_bests_values[0]
        self.update_bests()

        self.iter = 0
        self.is_running = True
        self.update_coef()

    def __str__(self):
        return f'[{self.iter}/{self.max_iter}] $w$:{self.w:.3f} - $c_1$:{self.c_1:.3f} - $c_2$:{self.c_2:.3f}'

    def next(self):
        if self.iter > 0:
            self.move_particles()
            self.update_bests()
            self.update_coef()

        self.iter += 1
        self.is_running = self.is_running and self.iter < self.max_iter
        return self.is_running

    def update_coef(self):
        if self.auto_coef:
            t = self.iter
            n = self.max_iter
            self.w = (0.4/n**2) * (t - n) ** 2 + 0.4
            self.c_1 = -3 * t / n + 3.5
            self.c_2 =  3 * t / n + 0.5

    def move_particles(self):
        # add inertia
        new_velocities = self.w * self.velocities
        # add cognitive component
        r_1 = np.random.random(self.N)
        r_1 = np.tile(r_1[:, None], (1, self.particles.shape[1]))
        new_velocities += self.c_1 * r_1 * (self.p_bests - self.particles)
        # add social component
        r_2 = np.random.random(self.N)
        r_2 = np.tile(r_2[:, None], (1, self.particles.shape[1]))
        g_best = np.tile(self.g_best[None], (self.N, 1))
        new_velocities += self.c_2 * r_2 * (g_best  - self.particles)

        self.is_running = np.sum(self.velocities - new_velocities) != 0

        # update positions and velocities
        self.velocities = new_velocities
        self.particles = self.particles + np.round(new_velocities).astype(int)

        # ensure the sum of the first 25 elements of each particle is equal to 30
        for i in range(self.particles.shape[0]):
            total = np.sum(self.particles[i, :24])
            if total != 30:
                self.particles[i, :24] = self.particles[i, :24] * 30 / total


        # set bounds for particles to brute force? Let the algorithm do its thing if possible todo compare
        #self.particles = np.clip(self.particles, 0, 10)


    def update_bests(self):
        fits = self.fitness_function(self.particles)

        for i in range(len(self.particles)):
            # update best personnal value (cognitive)
            if fits[i] < self.p_bests_values[i]:
                self.p_bests_values[i] = fits[i]
                self.p_bests[i] = self.particles[i]
                # update best global value (social)
                if fits[i] < self.g_best_value:
                    self.g_best_value = fits[i]
                    self.g_best = self.particles[i]



def calculate_price(particles):
    n_particles = particles.shape[0]

    # Allocate memory on the device
    particles_device = cuda.to_device(particles)
    prices_device = cuda.device_array(n_particles)

    # Calculate grid size
    threadsperblock = 32
    blockspergrid = (n_particles + (threadsperblock - 1)) // threadsperblock

    # Call CUDA kernel
    calculate_price_kernel[blockspergrid, threadsperblock](pricing, dishwasher, particles_device, prices_device)

    # Copy the result back to the host
    prices = prices_device.copy_to_host()

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
n_particles = 1000
particles = np.random.randint(0, 3, size=(n_particles, 25))  # Random initial offsets
velocities = np.random.uniform(-1, 10, size=(n_particles, 25))  # Random initial velocities

pso = PSO(particles, velocities, fitness_function)

start = time.time()

# Run PSO
while pso.next():
    pass

end = time.time()

print(f'Elapsed time: {end - start:.2f} seconds')

# Get best solution
best_particles = pso.g_best
best_price = pso.g_best_value

# Convert best particles to usage array for grid
ev_consumption = best_particles[:24]
dishwasher_offset = best_particles[24]

total_power_usage = np.zeros(24, dtype=float)
total_power_usage += ev_consumption
#total_power_usage[dishwasher_offset:dishwasher_offset + len(dishwasher)] += dishwasher

# Plot the total_power_usage
plt.bar(range(24), total_power_usage, label=f'Best Price: {round(best_price, 2)} EUR')
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
