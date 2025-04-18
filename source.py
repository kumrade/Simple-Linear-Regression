import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# ------------------------------
# Step 1: Simulate Synthetic Data
# ------------------------------
np.random.seed(0)
n = 500
true_intercept = 1.0
true_slope = 2.0
error_std = 3.0

# Independent variable (e.g., market returns)
X_raw = 10 * np.random.rand(n)

# Random noise
error_term = np.random.normal(0, error_std, n)

# Dependent variable (e.g., stock returns)
Y = true_intercept + true_slope * X_raw + error_term

# ------------------------------
# Step 2: Fit Linear Regression Model
# ------------------------------
model = LinearRegression(fit_intercept=True)
model.fit(X_raw.reshape(-1, 1), Y)

estimated_intercept = model.intercept_
estimated_slope = model.coef_[0]

print(f"Estimated Intercept: {estimated_intercept:.4f}")
print(f"Estimated Slope    : {estimated_slope:.4f}")

# ------------------------------
# Step 3: Visualize True Line vs Fitted Line
# ------------------------------
x_plot = np.linspace(0, 10, n)
y_true_line = true_intercept + true_slope * x_plot
y_pred_line = model.predict(x_plot.reshape(-1, 1))

plt.figure(figsize=(8, 5))
plt.scatter(X_raw, Y, color='steelblue', alpha=0.6, label='Data')
plt.plot(x_plot, y_true_line, color='green', linewidth=2.5, label='True Line')
plt.plot(x_plot, y_pred_line, color='red', linewidth=2.5, label='Fitted Line')
plt.title("True vs Fitted Regression Line")
plt.xlabel("Independent Variable (X)")
plt.ylabel("Dependent Variable (Y)")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# Step 4: Model Performance Metrics
# ------------------------------
r_squared = model.score(X_raw.reshape(-1, 1), Y)
correlation, _ = pearsonr(X_raw, Y)

print(f"R-squared: {r_squared:.4f}")
print(f"Square Root of R² (Correlation): {np.sqrt(r_squared):.4f}")
print(f"Pearson Correlation Coefficient: {correlation:.4f}")
