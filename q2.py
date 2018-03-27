#################################
#Author: Pawan Harendra Mishra
#EE671A: Neural Networks
#Question 2: Backpropagation with
#adaptive learning
#################################

from numpy import *
from matplotlib.pyplot import *
import os

#training data from csv file
trainData = genfromtxt("data12.csv", delimiter = ',')
#testing data from csv file
testData = genfromtxt("data22.csv", delimiter = ',')

#global parameters
lr = 0.001 #adaptive learning rate constant
epsilon = 2 #convergence
n, m = shape(trainData)
l0  = m - 2 #number of input layer nodes
l1 = 5 #number of first layer nodes
l2 = 2 #nuber of second layer nodes
#l3 = # #number of third layer nodes
#l4 = # #number of fourth layer nodes

#frobenius norm
def norm(x):
    return sqrt(sum(x**2))

#sigmoidal activation function
def sigmoid(x):
    return 1/(1 + exp(-x))

#function for training the network
def training():
    x0 = trainData[:,2:m] #input data
    yd = trainData[:,:2] #output data

    #random initialization of weights between -1 and 1
    w0 = 2*random.random((l0, l1)) - 1
    w1 = 2*random.random((l1, l2)) - 1

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
        J1e = dot(x1.T, d1)
        w1 = w1 + (lr*norm(e)/norm(J1e))*J1e

        d0 = x1*(1-x1)*dot(d1, w1.T)
        J0e = dot(x0.T, d0)
        w0 = w0 + (lr*norm(e)/norm(J0e))*J0e

        error.append(sum(e**2))

        if it%50 == 0:
            print it, error[-1]

        if error[-1] < epsilon or it > 100000:
            break

    print error[-1]
    return w0, w1, error, x0, yd

flag = True
if os.path.isfile("q2_w0.csv") and os.path.isfile("q2_w1.csv"):
    print "Pretrained weights exist. Do you want to use them?"
    response = raw_input("[Y/N]: ")
    if response == "Y" or response == "y":
        w0 = np.genfromtxt("q2_w0.csv", delimiter=",")
        w1 = np.genfromtxt ("q2_w1.csv", delimiter=",")
        flag = False

if flag:
    w0, w1, error, x0, yd = training()
    #save trained weights
    savetxt("q2_w0.csv", w0, delimiter = ',')
    savetxt("q2_w1.csv", w1, delimiter = ',')

    h0 = dot(x0, w0)
    x1 = sigmoid(h0)
    h1 = dot(x1, w1)
    x2 = sigmoid(h1)

    f1 = figure()
    ax1 = f1.add_subplot(1,1,1)
    ax1.plot(array(range(1, n+1)), yd[:, 0], color = 'g', label = "Vx Data")
    ax1.plot(array(range(1, n+1)), x2[:, 0], color = 'r', label = "Vx trained output")
    xlabel("Data Point")
    ylabel("Vx")
    ax1.legend()
    title("Question 2: Vx trained")
    savefig("q2_1.png")

    f2 = figure()
    ax2 = f2.add_subplot(1,1,1)
    ax2.plot(array(range(1, n+1)), yd[:, 1], color = 'g', label = "Vy Data")
    ax2.plot(array(range(1, n+1)), x2[:, 1], color = 'r', label = "Vy trained output")
    xlabel("Data Point")
    ylabel("Vy")
    ax2.legend()
    title("Question 2: Vy trained")
    savefig("q2_2.png")

    n = len(error)
    f3 = figure()
    ax3 = f3.add_subplot(1,1,1)
    ax3.plot(array(range(1, n+1)), array(error), color = 'r')
    xlabel("iterations")
    ylabel("error")
    title("Question 2: Error Vs. Iterations")
    savefig("q2_3.png")


n, m = shape(testData)
x0 = testData[:,2:m] #input data
yd = testData[:,:2] #output data
h0 = dot(x0, w0)
x1 = sigmoid(h0)
h1 = dot(x1, w1)
x2 = sigmoid(h1)

f4 = figure()
ax4 = f4.add_subplot(1,1,1)
ax4.plot(array(range(1, n+1)), yd[:, 0], color = 'g', label = "Vx Data")
ax4.plot(array(range(1, n+1)), x2[:, 0], color = 'r', label = "Vx tested output")
xlabel("Data Point")
ylabel("Vx")
ax4.legend()
title("Question 2: Vx tested")
savefig("q2_4.png")

f5 = figure()
ax5 = f5.add_subplot(1,1,1)
ax5.plot(array(range(1, n+1)), yd[:, 1], color = 'g', label = "Vy Data")
ax5.plot(array(range(1, n+1)), x2[:, 1], color = 'r', label = "Vy tested output")
xlabel("Data Point")
ylabel("Vy")
ax5.legend()
title("Question 2: Vy tested")
savefig("q2_5.png")
