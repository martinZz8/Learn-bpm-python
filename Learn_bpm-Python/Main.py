import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Net import Net


def trainNet(my_net, x_train, y_train, epoch):
    # Wyczyszczenie pliku z danymi uzyskanymi w wyniku uczenia
    open("Results_of_train.txt", "w").close()

    # Uczenie sieci
    file = open("Results_of_train.txt", "a")
    avg_rms = []
    for e in range(epoch):
        # Zapisuje wyniki ostatnich 5 epok
        if (e >= (epoch - 5)) and (epoch - 5 >= 0):
            saveEpochToFile(file, e + 1)
        elif epoch - 5 < 0:
            saveEpochToFile(file, e + 1)
        rms = []
        for r in range(len(x_train)):
            input_values = x_train[r]
            target_values = y_train[r]
            my_net.feedForward(input_values)
            my_net.backProp(target_values, rms)
            result_values = my_net.getResults()
            if (e >= (epoch - 5)) and (epoch - 5 >= 0):
                saveResultsToFile(file, r, result_values, target_values)
            elif epoch - 5 < 0:
                saveResultsToFile(file, r, result_values, target_values)
        avg_rms.append(sum(rms) / len(rms))
        print("Epoch: %d" % (e + 1))
    plt.plot(range(len(avg_rms)), avg_rms)
    file.close()
    printCurrentPercent(avg_rms, len(y_train[0]), "train net")
    plt.title("Wynik nauki sieci")
    plt.xlabel("Epoch")
    plt.ylabel("Avg RMS in epoch")
    plt.show()


def testNet(my_net, x_test, y_test, number_of_output_neurons):
    # Wyczyszczenie pliku z danymi uzyskanymi w wyniku testowania
    open("Results_of_test.txt", "w").close()

    # Testowanie sieci
    file = open("Results_of_test.txt", "a")
    rms = []
    for r in range(len(x_test)):
        input_values = x_test[r]
        target_values = y_test[r]
        my_net.feedForward(input_values)
        result_values = my_net.getResults()
        saveResultsToFile(file, r, result_values, target_values)
        # Obliczanie wlasnego RMS
        error = 0
        for i in range(len(result_values)):
            delta = target_values[i] - result_values[i]
            error += delta**2
        error /= number_of_output_neurons
        error = np.sqrt(error) # RMS
        rms.append(error)
    plt.plot(range(len(rms)), rms)
    file.close()
    printCurrentPercent(rms, len(y_test[0]), "test net")
    plt.title("Wynik testow sieci")
    plt.xlabel("Data row")
    plt.ylabel("RMS")
    plt.show()


def saveEpochToFile(file, epoch):
    file.write("Epoch: {}\n\n-\t-\t-\t-\t-\t-\n\n".format(epoch))


def saveResultsToFile(file, row, result_values, target_values):
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


# Wypisanie procentu jak dobrze uczy sie siec
def printCurrentPercent(rms, number_of_lln, name):
    current_percent = (1 - sum(rms) * np.sqrt(number_of_lln)/len(rms))*100
    print("Current percent of %s: %.2f\n" % (name, current_percent))


def main():
    dataset = pd.read_csv("winequality-red.csv", sep=';', decimal=",", dtype=np.float)
    # dataset = dataset.sort_values(by='quality')
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1:].values
    y = OneHotEncoder().fit_transform(y).toarray().astype(float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    topology = [len(x_train[0]), 25, 20, len(y_train[0])]
    epoch = 5000
    my_net = Net(topology)
    trainNet(my_net, x_train, y_train, epoch)
    testNet(my_net, x_test, y_test, len(y_test[0]))
    print()


main()
