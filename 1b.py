import numpy as np
import matplotlib.pyplot as plt
# Input data
years = np.array([2013, 2014, 2015, 2016, 2017])
advertising = np.array([100000, 120000, 140000, 160000, 220000])
sales = np.array([100000, 120000, 140000, 160000, 220000])
# Calculate necessary values
n = len(years)
mean_years = np.mean(years)
mean_advertising = np.mean(advertising)
mean_sales = np.mean(sales)
# Using the formulas for slope and intercept in simple linear regression
slope = np.sum((years - mean_years) * (sales - mean_sales)) / np.sum((years - mean_years)**2)
intercept = mean_sales - slope * mean_years
# Predict sales for the next year (assuming next year is 2018)
next_year = 2018
predicted_sales = intercept + slope * next_year

print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"Predicted Sales for {next_year}: {predicted_sales}")
# Plot the data points and regression line
plotfigure = plt.scatter(years, sales, label='Actual Sales')
plt.plot(years, intercept + slope * years, color='red', label='Regression Line')
plt.scatter(next_year, predicted_sales, color='green', marker='X', label=f'Predicted Sales for {next_year}')
# Add labels and legend
plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Linear Regression - Sales vs Year')
plt.legend()
# Show the plot
plt.show()