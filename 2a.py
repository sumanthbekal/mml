import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
dataset = pd.read_csv('HeightWeight.csv')

# Separate features (heights) and target (weights)
heights = dataset['Height'].values.reshape(-1, 1)
weights = dataset['Weight'].values

# Split dataset into training and testing sets
heights_train, heights_test, weights_train, weights_test = train_test_split(
    heights, weights, test_size=0.2, random_state=42)

# Initialize Linear Regression model
model = LinearRegression()

# Train the model
model.fit(heights_train, weights_train)

# Make predictions on test set
weights_pred = model.predict(heights_test)

# Print model coefficients and intercept
print("Coefficient (slope):", model.coef_[0])
print("Intercept:", model.intercept_)

# Get user input for height
user_height = float(input("Enter the height in inches: "))
user_height_reshaped = np.array([[user_height]])

# Predict weight for user input height
predicted_weight = model.predict(user_height_reshaped)[0]
print(f"Predicted weight for a height of {user_height} inches: {predicted_weight:.2f} pounds")

# Visualize regression line and actual data
plt.scatter(heights_test, weights_test, color='black', label='Actual data')
plt.plot(heights_test, weights_pred, color='blue', linewidth=3, label='Regression line')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Linear Regression: Height vs Weight')
plt.legend()
plt.show()
