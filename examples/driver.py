import numpy as np
import matplotlib.pyplot as plt
from src.linalg_interp import spline_function  

# 1. Load the data from text files
water_data = np.loadtxt('examples/water_density.txt')  # two columns: temp, density
air_data = np.loadtxt('examples/air_density.txt')      #two columns: temp, density

# Split columns
temp_water = water_data[:, 0]
dens_water = water_data[:, 1]
temp_air = air_data[:, 0]
dens_air = air_data[:, 1]

# 2. Define spline orders
orders = [1, 2, 3]

# 3. Generate spline functions for water and air
spline_funcs_water = [spline_function(temp_water, dens_water, order=o) for o in orders]
spline_funcs_air = [spline_function(temp_air, dens_air, order=o) for o in orders]

# 4. Create a fine grid for interpolation (100 equally spaced points)
temp_water_fine = np.linspace(temp_water.min(), temp_water.max(), 100)
temp_air_fine = np.linspace(temp_air.min(), temp_air.max(), 100)

# 5. Prepare plotting
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle("Water and Air Density Spline Interpolation", fontsize=16)

for i, order in enumerate(orders):
    # Water subplot
    ax_w = axes[i, 0]
    ax_w.plot(temp_water, dens_water, 'o', label='Data')
    ax_w.plot(temp_water_fine, spline_funcs_water[i](temp_water_fine), '-', label=f'Order {order} spline')
    ax_w.set_title(f"Water Density, Order {order}")
    ax_w.set_xlabel("Temperature")
    ax_w.set_ylabel("Density")
    ax_w.legend()
    ax_w.grid(True)
    
    # Air subplot
    ax_a = axes[i, 1]
    ax_a.plot(temp_air, dens_air, 'o', label='Data')
    ax_a.plot(temp_air_fine, spline_funcs_air[i](temp_air_fine), '-', label=f'Order {order} spline')
    ax_a.set_title(f"Air Density, Order {order}")
    ax_a.set_xlabel("Temperature")
    ax_a.set_ylabel("Density")
    ax_a.legend()
    ax_a.grid(True)

# 6. Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle

# 7. Save the figure
plt.savefig('examples/spline_density_plots.png')
plt.show()

