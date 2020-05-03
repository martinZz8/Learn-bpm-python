import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Net import Net


def saveEpochToFile(file_name, epoch):
    file = open(file_name, "a")
    file.write("Epoch: {}\n\n-\t-\t-\t-\t-\t-\n\n".format(epoch))
    file.close()


def saveResultsToFile(file_name, row, result_values, target_values):
    file = open(file_name, "a")
    file.write("\nData row: {}\n\nOutput:   ".format(row))
    for i in range(len(result_values)):
        if i != (len(result_values) - 1):
            file.write("%.5f, " % result_values[i])
        else:
            file.write("%.5f\n" % result_values[i])
    file.write("Expected: ")
    for i in range(len(target_values)):
        if i != (len(target_values) - 1):
            file.write("%.5f, " % target_values[i])
        else:
            file.write("%.5f\n" % target_values[i])
    file.close()


def main():
    dataset = pd.read_csv("winequality-red.csv", sep=';', decimal=",", dtype=np.float)
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1:].values
    y = OneHotEncoder().fit_transform(y).toarray().astype(float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    epoch = 10
    topology = [len(x_train[0]), len(x_train[0]), len(x_train[0]), len(y_train[0])]
    my_net = Net(topology)

    # Wyczyszczenie pliku z danymi
    open("Results.txt", "w").close()

    for e in range(epoch):
        saveEpochToFile("Results.txt", e+1)
        for r in range(len(x_train)):
            input_values = x_train[r]
            target_values = y_train[r]
            my_net.feedForward(input_values)
            my_net.backProp(target_values)
            result_values = my_net.getResults()
            saveResultsToFile("Results.txt", r, result_values, target_values)
    print()


main()
