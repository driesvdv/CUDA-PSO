import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_price(offset_1, offset_2):     
    dishw_price = np.sum(dishwasher * pricing[offset_1:offset_1 + len(dishwasher)])
    oven_price = np.sum(oven * pricing[offset_2:offset_2 + len(oven)])
    
    return dishw_price + oven_price
    

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

# Plot calculate price for different offsets (0..15) for device 1 and 2 in a 3D meshgrid
X = np.arange(0, 15, 1)
Y = np.arange(0, 15, 1)
meshgrid = np.meshgrid(X, Y)

# Create Z as a 2D array corresponding to the grid points
Z = np.array([[calculate_price(x, y) for x in X] for y in Y])

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

plt.show()