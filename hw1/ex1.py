import numpy as np
import matplotlib.pyplot as plt

N = 100
x = np.random.random(N)
y = 5*x**2 + 0.1*np.random.random(N)
plt.plot(x,y, 'o')
plt.show()

