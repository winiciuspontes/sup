"""
Escrevendo um codigo em python que aprenda a reconhecer digitos manuscritos usando 
Stochastic Gradient Descent Classifier (SGDClassifier) e o dataset MNIST.
"""
#imports 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt



"""
Na classe a seguir, temos que o tamanho contêm o numero de neuronios nas respectivas camadas,
sendo um objeto do tipo lista. 
Os bias e pesos no objeto são todos inicializados aleatoriamente, usando a distribuição Gaussiana com média 0 e variância 1.

"""


class Network():
    def __init__(self, tamanho):
        self.tamanho = tamanho
        self.n_layer = len(tamanho)
        self.bias = [np.random.randn(y, 1) for y in tamanho[1:]]
        self.weigths =  [np.random.randn(y, x) for x, y in zip(tamanho[:-1], tamanho[1:])]

    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))
    
    def feedfoward(self, a):
        for b, w in zip(self.bias, self.weigths):
            a = self.sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None): #Stochastic Gradient Descent - eta: learning rate
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                print(f'Epoch {j}: {self.evaluate(test_data)} / {n_test}')
            
            else:
                print(f'Epoch {j} complete')
        
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weigths]

        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.weigths = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weigths, nabla_w)]
        self.bias = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.bias, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weigths]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.bias, self.weigths):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.n_layer):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weigths[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)



redes1 = Network([3, 4, 1])


