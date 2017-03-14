import numpy as np
from matplotlib import pyplot as plt

upperbound = 10000
size_v_frac = np.ndarray(upperbound)

for n in range(1, upperbound):
    sample = np.random.randn(n)
    indices = np.random.random_integers(0, n - 1, n)

    bootstrap = sample[indices]
    s = set(bootstrap)
    frac = float(len(s)) / n

    size_v_frac[n] = frac

plt.plot(size_v_frac)
plt.title("Frequency of each element in a bootstrap sample approaches 1-1/e ~= 63.2%")
plt.xlabel("Number of standard normal random numbers")
plt.ylabel("Fraction of occurrence of the bootstrap samples in the original")
plt.show()
