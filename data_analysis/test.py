from multiprocessing import Pool, Lock, Array, Process, Value
import numpy as np
import matplotlib.pyplot as plt
from mesoplastic import utils
from typing import List, Tuple, Optional, Union
import random



# Generate some example data
data = np.random.rand(9, 10)  # 9x10 array of random values

# Create the 3x3 subplots
fig, axes = plt.subplots(3, 3)

# Plot the images and add colorbars
for i, ax in enumerate(axes.flat):
    img = ax.imshow(data[i].reshape((1, -1)), cmap='viridis')  # Reshape to 2D array
    cax = fig.add_axes([ax.get_position().x1, ax.get_position().y0, 0.02, ax.get_position().height])
    plt.colorbar(img, cax=cax)

# Set the range of colorbars based on the data
vmin = 0 # Minimum value for the column
vmax = [9, 15, 26]  # Maximum value for the column
for i in range(3):
    cbar = fig.colorbar(axes.flat[i*3].images[0], cax=cax)
    cbar.mappable.set_clim(vmin, vmax[i])  # Set the limits directly

# Adjust the layout and spacing of subplots
fig.tight_layout()

# Display the figure
plt.show()