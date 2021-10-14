#Wisconsin Breast Cancer  MLP Classifier
#WRITTEN BY MOHAMMAD ASADOLAHI
# Mohammad.E.Asadolahi@gmail.com
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class NueralNetwork:
    weights = []

    def __init__(self, layers):
        for item in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[item], layers[item - 1]))

    def sigmoid(self, number):
        return 1 / (1 + np.exp(-number))

    def deriviateOfSigmoid(self, number):
        return number * (1 - number)

    def printWeights(self):
        print(self.weights)

    def compile(self, trainSamples, trainLabels, testSamples, testlabels, epoches, learningRate=0.1):
        modelAccuracy = []
        for epoch in range(epoches):
            for item in range(len(trainSamples)):
                self.backpropagate(trainSamples[item], trainLabels[item], learningRate)
            modelAccuracy.append(self.modelEvaluation(testSamples, testlabels))
            # if (epoch % (epoches / 10)) == 0:
            #     print(f"epoch:{epoch} accuracy is{modelAccuracy[epoch]}")
            print(f"epoch: {epoch + 1} accuracy is {modelAccuracy[epoch - 1]}")
        plt.plot([x for x in modelAccuracy], label="Model Accuracy over epoches")
        plt.xlabel('x - Epoche')
        plt.ylabel('y - Accuracy ')
        plt.title('Model Accuracy over epoches')
        plt.legend()
        plt.show()

    def predict(self, sample):
        return self.feedForward(sample)[1][0]

    def modelEvaluation(self, testSamples, testlabels):
        correct = 0
        for index in range(len(testSamples)):
            prediction = self.predict(testSamples[index])
            if (prediction > 0.5 and testlabels[index] == 1) or (prediction <= 0.5 and testlabels[index] == 0):
                correct += 1
        return correct / len(testSamples)

    def feedForward(self, trainSample):
        input = np.copy(trainSample)
        outputs = []
        output = []
        for layer in self.weights:
            for node in layer:
                output.append(self.sigmoid(input.dot(node)))
            input = np.copy(output)
            outputs.append(input)
            output.clear()
        return outputs

    def erroCalculator(self, netOutputs, desiredOutput):
        outputError = (desiredOutput - netOutputs[len(self.weights) - 1]) * self.deriviateOfSigmoid(
            netOutputs[len(self.weights) - 1])
        hiddenLayerError = []
        for item in range(len(self.weights[0])):
            hiddenLayerError.append(
                (outputError[0] * self.weights[1][0][item]) * self.deriviateOfSigmoid(netOutputs[0][item]))
        return [np.array(hiddenLayerError), outputError]

    def backpropagate(self, input, desiredOutput, learningRate=0.1):
        netOutputs = self.feedForward(input)
        netErros = self.erroCalculator(netOutputs, desiredOutput)
        for bridge in range(9):
            for line in range(9):
                self.weights[0][bridge][line] = self.weights[0][bridge][line] + (
                        learningRate * ((input[line]) * netErros[0][line]))
        for line in range(9):
            self.weights[1][0][line] = self.weights[1][0][line] + ((learningRate * (netOutputs[0][line])) * netErros[1])


# fetching records from storage and preprocessing-------------------------------------------------
records = []
with open('./cancer.data') as f:
    for line in f.readlines():
        records.append(np.fromstring(line.rstrip('\n'), dtype=float, sep=","))
samples = pd.DataFrame(records, columns=["id", "Clump Thickness", "Cell Size Uniformity", "Cell Shape Uniformity",
                                         "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei",
                                         "Bland Chromatin", "Normal Nucleoli", "Mitoses", "class"])

print(samples.head(5))  # show top 5 fetch records

samples = samples.drop("id", axis=1)  # remove id witch is useless for classification

labels = samples["class"]  # transfer labels to another dataframe
samples = samples.drop("class", axis=1)  # remove class from features vector

labels[labels == 2] = 0  # "replace 2 with 0 for benign cancer"

labels[labels == 4] = 1  # "replace 4 with 1 for malignant cancer"



# replacing missig values with the mean of column--------------------------------------------------
for each in samples.columns:
    samples[each] = samples[each].replace(to_replace=0,
                                          value=int(samples[each].mean()))

# scaling all numeral features between [0 to 1] ----------------------------------------------------
for each in samples.columns:
    samples[each] = samples[each] / samples[each].max()

# split samples into train and test samples
trainSamples = (samples[0:450]).to_numpy()  # 450 sample to train the model ,almost 65% of all records
trainLabels = (labels[0:450]).to_numpy()

testSamples = (samples[450:699]).to_numpy()  # 250 sample to test the model ,almost 35% of all records
testlabels = (labels[450:699]).to_numpy()

# define NN model with 9 input neurons in input layer and 9 dense neurons in hidden layer and 1 neuron in output layer
nueralNetwork = NueralNetwork([9, 9, 1])

# train the model with train samples and their label and evaluate the model with test pationts at each step
nueralNetwork.compile(trainSamples, trainLabels, testSamples, testlabels, 100, 0.1)

# using our MLP to predict class of our test samples
for index in range(len(testSamples)):
    prediction = nueralNetwork.predict(testSamples[index])
    if (prediction > 0.5):
        prediction = 1
    else:
        prediction = 0
    print(f"sample: {testSamples[index]}  predicted class: {prediction}  real calss: {testlabels[index]}")
