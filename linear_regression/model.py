import numpy as np
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

def mse(y_true, y_pred):
    num_samples = len(y_true)
    mse = np.sum((y_true - y_pred) ** 2) / num_samples
    return mse

class LinearRegression:
    def __init__(self, learning_rate, iterations):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss = []
        
    def fit(self, X, y):
        num_samples, num_fetures = X.shape    # num_samples would be rows, num_features would be columns
        self.weights = np.zeros(num_fetures)  ## Creates array of zeros with length = number of features
        self.bias = 0.0
        
        for i in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = (2 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / num_samples) * np.sum(y_pred - y)
            
            self.weights = self.weights - (self.lr * dw)  # this is from the grdient decent equation
            self.bias = self.bias - (self.lr * db)
            
            # calculate the loss, also I defined a function to calculate MSE(Mean Squared Error)
            loss = mse(y, y_pred)
            self.loss.append(loss)
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    

# Testing the model

X = np.array([[2], [4], [6], [8]])
y = np.array([1, 2, 3, 4])

model = LinearRegression(learning_rate=0.01, iterations=2000)
model.fit(X, y)
pred = model.predict(X)
print(f"Prediction: {pred}")

# loss
loss = model.loss
print(f"Loss: {loss}")


# plotting the loss curve
plt.figure(figsize=(8, 5))
plt.plot(model.loss, label = "MSE Loss")
plt.title("Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()