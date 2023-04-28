import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from PIL import Image


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
epsilon = 1.1

# Calculate exact and WKB solutions
y_exact_values = y_exact(omega=x, epsilon=epsilon)
y_wkb_values = wkb(epsilon=epsilon, omega=x)

# Calculate the relative error
rel_error = (y_wkb_values - y_exact_values) / y_exact_values


# Create a list to store frames
frames = []

# Define the range and step size for epsilon
epsilon_range = np.linspace(2, 0.01, 50)

# Loop through epsilon values
for epsilon in epsilon_range:
    # Calculate exact and WKB solutions
    y_exact_values = y_exact(omega=x, epsilon=epsilon)
    y_wkb_values = wkb(epsilon=epsilon, omega=x)

    # Plot the exact and WKB solutions
    plt.plot(x, y_exact_values, 'black', label=f'Exact solution with epsilon = {epsilon:.2f}')
    plt.plot(x, y_wkb_values, 'blue', label=f'WKB solution with epsilon = {epsilon:.2f}')

    # Set the plot limits
    plt.xlim([0, x_range])
    plt.ylim([-2, 2])

    # Add legend
    plt.legend(loc='upper left')

    # Draw the canvas and convert it to a PIL image
    plt.draw()
    plt.pause(0.1)
    image = Image.frombytes('RGB', plt.gcf().canvas.get_width_height(), plt.gcf().canvas.tostring_rgb())
    frames.append(image)

    # Clear the plot for the next frame
    plt.clf()

# Save the frames as a GIF
frames[0].save('epsilon_variation.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)

plt.close()
