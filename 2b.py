import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('HeightWeight.csv')
heights = dataset['Height'].values
weights = dataset['Weight'].values
mean_height = np.mean(heights)
mean_weight = np.mean(weights)
numerator = np.sum((heights - mean_height) * (weights - mean_weight))
denominator = np.sum((heights - mean_height)**2)
slope = numerator / denominator
intercept = mean_weight - slope * mean_height
print("Slope (m):", slope)
print("Intercept (c):", intercept)
user_height_inches = float(input("Enter the height in inches: "))
predicted_weight = slope * user_height_inches + intercept
print(f"Predicted weight for a height of {user_height_inches} inches: {predicted_weight:.2f} pounds")
plt.scatter(heights, weights, color='black', label='Actual data')
plt.plot(heights, slope * heights + intercept, color='blue', linewidth=3, label='Regression line')
plt.scatter(user_height_inches, predicted_weight, color='red', marker='x', label='Predicted weight')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.title('Linear Regression: Height vs Weight')
plt.legend()
plt.show()