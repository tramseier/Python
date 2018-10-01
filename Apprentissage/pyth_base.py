# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 14:23:00 2018

@author: toto
"""

#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt


#tests
p = np.linspace(-2,2,5)
#mx,my = np.meshgrid(p,p)
#newm =mx -my
#nnm = np.abs(newm)



nn = np.arange(40)
ssss = sum(20>i>5 for i in nn)

f = lambda u: u * (u>=0) * (u<=1) + (u>1) # CDF
Finv = lambda u: u * (u>=0) * (u<=1) + (u>1) # Inverse CDF
qq = 3

y = np.array([[2,2],[2,2]])


w = np.array([1 ,2]);
wt = w.T
ress = np.dot(y,w)

#mygenerator = (x*x for x in range(3))
#for i in mygenerator:
#    print(i)
    
def createGenerator():
   mylist = range(3)
   for i in mylist:
       yield i*i,2*i

mygenerator = createGenerator() # create a generator
print("test generator",mygenerator) # mygenerator is an object!
for i,j in mygenerator:
    print(i,j)
    

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
y = np.ones((50,1));
tx = np.ones((50,2));
www = np.array([2,2])

for yy,txx in batch_iter(y,tx,10):
    print("ouais",yy,txx)
    
#def compute_stoch_gradient(y, tx, w):
#    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
#    # ***************************************************
#    # INSERT YOUR CODE HERE
#    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
#    # ***************************************************
#    grad = 0;
#    for minibatch_y, minibatch_tx in batch_iter(y, tx, 10):
#        temp = np.reshape(np.dot(minibatch_tx,www),(10,1))
#        grad += np.dot(minibatch_tx.T,(temp - minibatch_y));
#    return grad
#
#stoch = compute_stoch_gradient(y,tx,w).T
#pp = np.reshape(np.dot(tx,www),(50,1))
#temp = (pp - y)
#gradn = np.dot(tx.T,temp);
    
def compute_gradient(y, tx, w):
    e = (y-np.dot(tx,w));
    N = np.shape(y)[0];
    grad = (-1/N)*tx.T@e;
    return grad

grrrad = compute_gradient(y,tx,w)

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
#    grad = np.dot(tx.T,(np.dot(tx,w) - y))
    e = y-tx@w;
    loss = np.sum(e**2);
    grad = (-tx.T)*(e);
    return loss,grad


gg = compute_stoch_gradient(y,tx,w)

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w];
    losses = [];
    w = initial_w;
    
    for n_iter in range(max_iters):
        grad = compute_gradient(y,tx,w);
        loss = compute_loss(y,tx,w);
        w = w - gamma*grad ;
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
fr = y-np.dot(tx,w) != np.zeros()

def compute_loss(y, tx, w):
    """Calculate the loss.
    
    You can calculate the loss using mse or mae.
    """
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    # ***************************************************
    N = np.shape(y)[0];
    loss = (1./(2*N))*(np.linalg.norm(np.dot(tx,w) - y))**2;
    return loss

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient descent.h
    # ***************************************************
    w = initial_w;
    ws = []
    losses = [];
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        w = w -gamma*compute_stoch_gradient(minibatch_y,minibatch_tx,w);
        loss = compute_loss(y,tx,w);
        losses.append(loss)
        ws.append(w)

    return losses, ws

#def grad_pab(y,tx,w):
#    w.reshape((-1,1))
max_iters = 50
gamma = 0.7
batch_size = 13
e = (y-np.dot(tx,w));
N = np.shape(y)[0];
gradzer = (-1/N)*tx.T@e;

# Initialization
w_initial = np.array([0, 0])
looses,wss = stochastic_gradient_descent(y, tx, w_initial, batch_size, max_iters, gamma)
