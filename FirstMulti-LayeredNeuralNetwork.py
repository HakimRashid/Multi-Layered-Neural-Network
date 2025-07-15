import os
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    
    def __init__(self, weights_file):
        self.weight_file = weights_file
        self.learning_rate = 0.1
        if not os.path.exists(self.weight_file):
            self.weights_L1 = np.random.randn(4, 2) * 0.001
            self.weights_L2 = np.random.randn(3, 4) * 0.001
            self.weights_LOut = np.random.randn(3, 1) * 0.001
            self.bias_L1 = np.random.randn(4)
            self.bias_L2 = np.random.randn(3)
            self.bias_LOut = np.zeros(1)
            np.savez(self.weight_file, weights_L1=self.weights_L1, weights_L2=self.weights_L2, weights_LOut=self.weights_LOut, bias_L1=self.bias_L1, bias_L2=self.bias_L2, bias_LOut=self.bias_LOut)
        else:
            data = np.load(self.weight_file)
            self.weights_L1 = data['weights_L1']
            self.weights_L2 = data['weights_L2']
            self.weights_LOut = data['weights_LOut']
            self.bias_L1 = data['bias_L1']
            self.bias_L2 = data['bias_L2']
            self.bias_LOut = data['bias_LOut']
            
    @staticmethod
    def ReLU(x: float):
        return np.maximum(0, x)
    @staticmethod
    def sigmoid(x: float):
        return 1/(1+np.exp(-x))
    
    def forwardpass(self, input):
        z_L1 = np.dot(input, self.weights_L1.T) + self.bias_L1
        a_L1 = np.tanh(z_L1)
        
        z_L2 = np.dot(a_L1, self.weights_L2.T) + self.bias_L2
        a_L2 = np.tanh(z_L2)
        
        z_out = np.dot(a_L2, self.weights_LOut) + self.bias_LOut
        a_out = NeuralNetwork.sigmoid(z_out)
        return a_out, z_out, a_L2, z_L2, a_L1, z_L1



    def backpropagation(self, target, inputs):
        scores = self.forwardpass(inputs)
        prediction =  np.clip(scores[0], 1e-7, 1 - 1e-7)
        loss = np.mean(-(target * np.log(prediction) + (1-target) * np.log(1-prediction)))
        print(f"Loss={loss}")
        
        error = prediction - target
        blames_out = np.outer(scores[2], error)
        
        d_L2 = (error * self.weights_LOut.flatten()) * (1- scores[2] ** 2)
        blames_L2 = np.outer(d_L2, scores[4])
        
        d_L1 = np.dot(self.weights_L2.T, d_L2) * (1- scores[5] ** 2)
        blames_L1 = np.outer(d_L1, inputs)
        
        self.weights_LOut -= blames_out * self.learning_rate
        self.weights_L1 -= blames_L1 * self.learning_rate
        self.weights_L2 -= blames_L2 * self.learning_rate
        
        self.bias_LOut -= self.learning_rate * error
        self.bias_L2   -= self.learning_rate * d_L2
        self.bias_L1   -= self.learning_rate * d_L1
        
        
    
    def predict(self, input):
        return int(np.round(self.forwardpass(input)[0])[0])
    
    def train(self, target, inputs):
        for epoch in  range(1000):
            for i in range(len(inputs)):
                self.backpropagation(target[i], inputs[i])
            
            if epoch % 100 == 0:
                predictions = np.array([self.predict(x) for x in inputs]).flatten()
                plt.plot(predictions, label='Prediction')
                plt.plot(target, label='Target')
                plt.legend()
                plt.show()
                np.savez(self.weight_file, weights_L1=self.weights_L1, weights_L2=self.weights_L2, weights_LOut=self.weights_LOut, bias_L1=self.bias_L1, bias_L2=self.bias_L2, bias_LOut=self.bias_LOut)


i = input("Would you like to train the Network(Y/N):")
nn = NeuralNetwork('weights.npz')
train = True if i == 'Y' else False

if train:
    training_data = np.array([[0,0],
                 [0,1],
                 [1,0],
                 [1,1]])
    target = np.array([0, 1, 1, 0])
    nn.train(target, training_data)
else:
    a = int(input("A: "))
    b = int(input("B: "))
    print(f"output= {nn.predict([a, b])}")