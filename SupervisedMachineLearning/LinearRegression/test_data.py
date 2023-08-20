import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from linear_regression import LinearRegression


X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

lr_model = LinearRegression(lr=0.01)
lr_model.fit(X_train, y_train)
predicted = lr_model.predict(X_test)

mse_value = LinearRegression.mse(y_test, predicted)
r2_score_value = LinearRegression.r2_score(y_test, predicted)
print(f"Mean Squared Error: {mse_value}")
print(f"R2 Score: {r2_score_value})")

y_pred_line = lr_model.predict(X)
cmap = plt.get_cmap("plasma")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10, label="Train Data")
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10, label="Test Data")
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.title("Linear Regression Prediction")
plt.legend()
plt.show()
