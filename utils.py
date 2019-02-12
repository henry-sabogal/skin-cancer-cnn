import numpy as np
from math import sqrt

def weigh_initialization(D,K):
    #initialize parameteres randomly
    W = np.zeros((K,D+1))
    
    for ii in range(K):
        W[ii] = W[ii] + np.random.randn(D+1) / sqrt(D+1)
    return W

def scores(W, x):
    x = x.flatten()
    x = np.append(x, 1)
    scores = np.dot(W,x.T)
    return scores

def SVM_loss(W, X, y, reg, delta):
    loss = 0.0
    num_classes_gradient = 0.0
    grad = np.zeros(W.shape)
    correct_class_index = 0
    for ii in range(len(X)):
        s = scores(W, X[ii])
        for jj in range(len(s)):
            if jj == y[ii]:
                continue
                
            margin = s[jj] - s[y[ii]] + delta
            if margin > 0:
                loss += margin
                num_classes_gradient += 1
                    
    loss /= len(X)
    loss += reg * np.sum(W*W)
     
    return loss

def gradient(W, X, y, reg, delta):
    grad = np.zeros(W.shape)
    h = 0.00001
    
    fx = SVM_loss(W, X, y, reg, delta)
    
    it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_value = W[ix]
        W[ix] = old_value + h
        fxh = SVM_loss(W, X, y, reg, delta)
        W[ix] = old_value
        
        grad[ix] = (fxh - fx) / 2*h
        it.iternext()
    
    return grad

def train(X, y, learning_rate, reg, num_iters):
    loss_history = []
    
    return loss_history