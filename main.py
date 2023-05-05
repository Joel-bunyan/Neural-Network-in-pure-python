import numpy as np
class Dense():
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.bias = np.random.randn(output_size, 1) * 0.1
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)
#X = np.reshape([[1], [5], [8], [10]], (4, 1, 1))
#Y = np.reshape([[3], [11], [17], [30]], (4, 1, 1)) 
X = np.array(range(10))
Y = (X*5) + 0.5

network = [Dense(1,2),
          Dense(2,1)]

outputs = []
for i in range(100):
  error =0
  for x,y in zip(X,Y):
    output = x
    for layer in network:
      output = layer.forward(output)
    print(float(output))
    outputs.append(output)
    error += mse(y,output)
  
  grad = mse_prime(y,output)
  for layer in reversed(network):
    grad = layer.backward(grad,0.0001)
  
  error/= 10
  print("epoch no",i+1,"error: ",error)