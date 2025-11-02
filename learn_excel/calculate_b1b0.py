import matplotlib.pyplot as plt
import numpy as np

# Data
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T
y = np.array([[2, 4, 3, 6, 9, 12, 13, 15, 18, 20]]).T

# Function to calculate b0 and b1
def calculate_b1b0(x, y):
    # Calculate averages
    xbar = np.mean(x)
    ybar = np.mean(y)
    x2bar = np.mean(x ** 2)
    xybar = np.mean(x * y)

    # Calculate b1 and b0
    b1 = (xybar - xbar * ybar) / (x2bar - (xbar ** 2))
    b0 = ybar - b1 * xbar
    return b1, b0

# Calculate b1, b0
b1, b0 = calculate_b1b0(x, y)
print("b1 =", b1)
print("b0 =", b0)

# Predicted y values
y_predicted = b0 + b1 * x
print("y_predicted =", y_predicted)

# Visualization (optional)
plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_predicted, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Simple Linear Regression Example')
plt.show()


# Visualize data
def showGraph(x, y, y_predicted, title="", xLabel="", yLabel=""):
    plt.figure(figsize=(14, 8))
    plt.plot(x, y, 'r-o', label='value sample')
    plt.plot(x, y_predicted, 'b-*', label='predicted value')

    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    # Mean of y
    ybar = np.mean(y)

    # Draw mean line
    plt.axhline(ybar, linestyle='--', linewidth=4, label='mean')

    # Set axis limits
    plt.axis([x_min * 0.95, x_max * 1.05, y_min * 0.95, y_max * 1.05])

    # Labels and title
    plt.xlabel(xLabel, fontsize=16)
    plt.ylabel(yLabel, fontsize=16)
    plt.text(x_min, ybar + 1, s='mean', fontsize=16)
    plt.legend(fontsize=15)
    plt.title(title, fontsize=20)
    plt.show()


# Call the function
showGraph(
    x, y, y_predicted,
    title='Y values corresponding to X',
    xLabel='X values',
    yLabel='Y values'
)
