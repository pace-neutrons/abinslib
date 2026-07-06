"""Sample plot
===========

A simple demo to make sure docs are building correctly.
"""

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 101)
y = np.sin(2 * x)

ax.plot(x, y)
plt.show()
