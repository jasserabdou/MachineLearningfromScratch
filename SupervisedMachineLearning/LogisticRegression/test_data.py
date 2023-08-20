import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from logistic_regression import LogisticRegression

bc = load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

lr_model = LogisticRegression(lr=0.01, n_iters=1000)
lr_model.fit(X_train, y_train)
predicted = lr_model.predict(X_test)

mse_value = lr_model.mse(y_test, predicted)
r2_score_value = lr_model.r2_score(y_test, predicted)
print(f"Mean Squared Error: {mse_value}")
print(f"R2 Score: {r2_score_value}")
print("Accuracy:", LogisticRegression.accuracy(y_test, predicted))
