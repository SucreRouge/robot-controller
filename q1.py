#################################
#Author: Pawan Harendra Mishra
#EE671A: Neural Networks
#Question 1: Backpropagation with
#momentum
#################################

from numpy import *
from matplotlib.pyplot import *
import os

#training data from csv file
trainData = genfromtxt("data12.csv", delimiter = ',')
#testing data from csv file
testData = genfromtxt("data22.csv", delimiter = ',')

#global parameters
lr = 0.001 #learning rate
a = 0.0005 #momentum constant
epsilon = 2 #convergence
n, m = shape(trainData)
l0  = m - 2 #number of input layer nodes
l1 = 5 #number of first layer nodes
l2 = 2 #nuber of second layer nodes
#l3 = # #number of third layer nodes
#l4 = # #number of fourth layer nodes

#sigmoidal activation function
def sigmoid(x):
    return 1/(1 + exp(-x))

#function for training the network
def training():
    x0 = trainData[:,2:m] #input data
    yd = trainData[:,:2] #output data

    #random initialization of weights between -1 and 1
    w0 = 2*random.random((l0, l1)) - 1
    w0Prev = zeros((l0, l1))
    w1 = 2*random.random((l1, l2)) - 1
    w1Prev = zeros((l1, l2))

    it = 0
    error = []

    while True:
        #forward propagation
        it += 1
        h0 = dot(x0, w0)
        x1 = sigmoid(h0)
        h1 = dot(x1, w1)
        x2 = sigmoid(h1)
        e = yd - x2

        #back propagation
        d1 = e*x2*(1-x2)
        w1Curr = w1
        w1 = w1 + lr*dot(x1.T, d1) + a*(w1 - w1Prev)
        w1Prev = w1Curr

        d0 = x1*(1-x1)*dot(d1, w1.T)
        w0Curr = w0
        w0 = w0 + lr*dot(x0.T, d0) + a*(w0 - w0Prev)
        w0Prev = w0Curr

        error.append(sum(e**2))

        if it%50 == 0:
            print it, error[-1]

        if error[-1] < epsilon or it > 100000:
            break

    print error[-1]
    return w0, w1, error, x0, yd

flag = True
if os.path.isfile("q1_w0.csv") and os.path.isfile("q1_w1.csv"):
    print "Pretrained weights exist. Do you want to use them?"
    response = raw_input("[Y/N]: ")
    if response == "Y" or response == "y":
        w0 = np.genfromtxt("q1_w0.csv", delimiter=",")
        w1 = np.genfromtxt ("q1_w1.csv", delimiter=",")
        flag = False

if flag:
    w0, w1, error, x0, yd = training()
    #save trained weights
    savetxt("q1_w0.csv", w0, delimiter = ',')
    savetxt("q1_w1.csv", w1, delimiter = ',')

    h0 = dot(x0, w0)
    x1 = sigmoid(h0)
    h1 = dot(x1, w1)
    x2 = sigmoid(h1)

    f1 = figure()
    ax1 = f1.add_subplot(1,1,1)
    ax1.plot(array(range(1, n+1)), yd[:, 0], color = 'r', label = "Vx Data")
    ax1.plot(array(range(1, n+1)), x2[:, 0], color = 'b', label = "Vx trained output")
    xlabel("Data Point")
    ylabel("Vx")
    ax1.legend()
    title("Question 1: Vx trained")
    savefig("q1_1.png")

    f2 = figure()
    ax2 = f2.add_subplot(1,1,1)
    ax2.plot(array(range(1, n+1)), yd[:, 1], color = 'r', label = "Vy Data")
    ax2.plot(array(range(1, n+1)), x2[:, 1], color = 'b', label = "Vy trained output")
    xlabel("Data Point")
    ylabel("Vy")
    ax2.legend()
    title("Question 1: Vy trained")
    savefig("q1_2.png")

    n = len(error)
    f3 = figure()
    ax3 = f3.add_subplot(1,1,1)
    ax3.plot(array(range(1, n+1)), array(error), color = 'r')
    xlabel("iterations")
    ylabel("error")
    title("Question 1: Error Vs. Iterations")
    savefig("q1_3.png")


n, m = shape(testData)
x0 = testData[:,2:m] #input data
yd = testData[:,:2] #output data
h0 = dot(x0, w0)
x1 = sigmoid(h0)
h1 = dot(x1, w1)
x2 = sigmoid(h1)

f4 = figure()
ax4 = f4.add_subplot(1,1,1)
ax4.plot(array(range(1, n+1)), yd[:, 0], color = 'r', label = "Vx Data")
ax4.plot(array(range(1, n+1)), x2[:, 0], color = 'b', label = "Vx tested output")
xlabel("Data Point")
ylabel("Vx")
ax4.legend()
title("Question 1: Vx tested")
savefig("q1_4.png")

f5 = figure()
ax5 = f5.add_subplot(1,1,1)
ax5.plot(array(range(1, n+1)), yd[:, 1], color = 'r', label = "Vy Data")
ax5.plot(array(range(1, n+1)), x2[:, 1], color = 'b', label = "Vy tested output")
xlabel("Data Point")
ylabel("Vy")
ax5.legend()
title("Question 1: Vy tested")
savefig("q1_5.png")
