import random
import math

#input array and expected output array
inputs = [[0,0],[0,1],[1,0],[1,1]]
ideal_outputs = [0, 1, 1, 0]
print("inputs: ")
print(inputs)
print("ideal outputs: ")
print(ideal_outputs)

#randomly initialize weights between -10 and 10
weight_I1toH1 = random.uniform(-10,10)
print("weight_I1toH1: " + str(weight_I1toH1))
weight_I1toH2 = random.uniform(-10,10)
print("weight_I1toH2: " + str(weight_I1toH2))
weight_I2toH1 = random.uniform(-10,10)
print("weight_I2toH1: " + str(weight_I2toH1))
weight_I2toH2 = random.uniform(-10,10)
print("weight_I2toH2: " + str(weight_I2toH2))
weight_B1toH1 = random.uniform(-10,10)
print("weight_B1toH1: " + str(weight_B1toH1))
weight_B1toH2 = random.uniform(-10,10)
print("weight_B1toH2: " + str(weight_B1toH2))
weight_H1toOut = random.uniform(-10,10)
print("weight_H1toOut: " + str(weight_H1toOut))
weight_H2toOut = random.uniform(-10,10)
print("weight_H2toOut: " + str(weight_H2toOut))
weight_B2 = random.uniform(-10,10)
print("weight_B2: " + str(weight_B2))

def sigmoid_function(num):
  return 1/(1 + (math.e)**(num))

def sigmoidDerivative(num):
  val = sigmoid_function(num)
  return val * (1 - val)

def printValues(startnode, endnode, currentiteration, weight, gradient):
  print("From node " + str(startnode) + " to node " + str(endnode) + " on iteration " + str(currentiteration) + ", the weight is " + str(weight) + " and the gradient is " + str(gradient)) 

def changeWeights(temporaryIteration, gradient):
  if temporaryIteration == 0:
    return learningRate * gradient
  if temporaryIteration != 0:
    return ((learningRate * gradient) + momentum * changeWeights(temporaryIteration - 1, gradient))


averageError = 1
currentIteration = 0
learningRate = 0.7
momentum = 0.3
#batch training
while (abs(averageError) > 0.05):
  errorSum = 0
  temporaryIteration = 0
  for i in range(0,4):
    input1 = inputs[i][0]
    input2 = inputs[i][1]
    ideal = ideal_outputs[i]
    #hidden node H1
    NetH1 = weight_I1toH1 * input1 + weight_I2toH1 * input2 + weight_B1toH1
    H1Output = sigmoid_function(NetH1)
    #hidden node H2
    NetH2 = weight_I1toH2 * input1 + weight_I2toH2 * input2 + weight_B1toH2
    H2Output = sigmoid_function(NetH2)
    #output node
    NetO1 = weight_H1toOut * H1Output + weight_H2toOut * H2Output + weight_B2
    output = sigmoid_function(NetO1)
    #error
    error = ideal - output
    errorSum += abs(error)
    averageError = errorSum/(temporaryIteration + 1)
    print("average error: " +str(averageError))
    #calculate node deltas
    NDeltaH1 = -1 * sigmoidDerivative(NetH1) * error
    NDeltaH2 = -1 * sigmoidDerivative(NetH2) * error
    NDeltaOutput = -1 * sigmoidDerivative(NetO1) * error

    #gradients for this iteration, sum to get batch gradient
    Gradient_I1toH1 = input1 * NDeltaH1
    Gradient_I1toH2 = input1 * NDeltaH2
    Gradient_I2toH1 = input2 * NDeltaH1
    Gradient_I2toH2 = input2 * NDeltaH2
    Gradient_B1toH1 = NDeltaH1
    Gradient_B1toH2 = NDeltaH2
    Gradient_H1toOutput = H1Output * NDeltaOutput
    Gradient_H2toOutput = H2Output * NDeltaOutput
    Gradient_B2toOutput = NDeltaOutput
    #batch here
    batchGradient = Gradient_I1toH1 + Gradient_I1toH2 + Gradient_I2toH1 + Gradient_I2toH2 + Gradient_B1toH1 + Gradient_B1toH2 + Gradient_H1toOutput + Gradient_H2toOutput + Gradient_B2toOutput

    #print values using printValues
    printValues("I1", "H1", currentIteration, weight_I1toH1, Gradient_I1toH1)
    printValues("I1", "H2", currentIteration, weight_I1toH2, Gradient_I1toH2)
    printValues("I2", "H1", currentIteration, weight_I2toH1, Gradient_I2toH1)
    printValues("I2", "H2", currentIteration, weight_I2toH2, Gradient_I2toH2)
    printValues("B1", "H1", currentIteration, weight_B1toH1, Gradient_B1toH1)
    printValues("B1", "H2", currentIteration, weight_B1toH2, Gradient_B1toH2)
    printValues("H1", "O1", currentIteration, weight_H1toOut, Gradient_H1toOutput)
    printValues("H2", "O1", currentIteration, weight_H2toOut, Gradient_H2toOutput)
    printValues("B2", "O1", currentIteration, weight_B2, Gradient_B2toOutput)
    print()

    #adjust weights
    weight_I1toH1 += changeWeights(temporaryIteration, batchGradient)
    weight_I1toH2 += changeWeights(temporaryIteration, batchGradient)
    weight_I2toH1 += changeWeights(temporaryIteration, batchGradient)
    weight_I2toH2 += changeWeights(temporaryIteration, batchGradient)
    weight_B1toH1 += changeWeights(temporaryIteration, batchGradient)
    weight_B1toH2 += changeWeights(temporaryIteration, batchGradient)
    weight_H1toOut += changeWeights(temporaryIteration, batchGradient)
    weight_H2toOut += changeWeights(temporaryIteration, batchGradient)
    weight_B2 += changeWeights(temporaryIteration, batchGradient)
    
    temporaryIteration += 1
    currentIteration += 1
    