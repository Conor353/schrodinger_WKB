import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

# Create linearly spaced numbers
x_range = 3
num_steps = 500
x = np.linspace(-x_range, x_range, num_steps)

# Define the exact solution function
def y_exact(omega, epsilon):
    numerator = (sp.jv(1/6, 8/(3*epsilon)) * sp.jv(-1/6, (omega**3)/(3*epsilon))
                 - sp.jv(-1/6, 8/(3*epsilon)) * sp.jv(1/6, (omega**3)/(3*epsilon)))

    denominator = (sp.jv(-1/6, 1/(3*epsilon)) * sp.jv(1/6, 8/(3*epsilon))
                   - sp.jv(-1/6, 8/(3*epsilon)) * sp.jv(1/6, 1/(3*epsilon)))

    D = numerator / denominator
    y = (omega**0.5) * D

    return y

# Define the WKB approximation function
def wkb(epsilon, omega):
    p = epsilon
    i = complex(0, 1)

    big_term = (1 - i) / (np.sqrt(2) * (np.exp((16 * i) / (3 * p)) - np.exp((2 * i) / (3 * p))))

    c_1 = -np.exp(i / (3 * p)) * big_term
    c_2 = np.exp((17 * i) / (3 * p)) * big_term

    coef = complex(1, 1) / np.sqrt(2)
    exponent = complex(0, 1) * (omega**3) / (p * 3)

    expon_1 = np.exp(exponent)
    expon_2 = np.exp(-exponent)

    y = coef * c_1 * (1 / omega) * expon_1 + coef * c_2 * (1 / omega) * expon_2

    return y

# Set epsilon value
epsilon = 0.5

# Calculate exact and WKB solutions
y_exact_values = y_exact(omega=x, epsilon=epsilon)
y_wkb_values = wkb(epsilon=epsilon, omega=x)

# Calculate the relative error
rel_error = (y_wkb_values - y_exact_values) / y_exact_values

# Configure the plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.xlim([0, x_range])
plt.ylim([-2, 2])

# Plot the exact and WKB solutions
plt.plot(x, y_exact_values, 'black', label=f'Exact solution with epsilon = {epsilon}')
plt.plot(x, y_wkb_values, 'blue', label=f'WKB solution with epsilon = {epsilon}')

# Add legend and display the plot
plt.legend(loc='upper left')
plt.show()
