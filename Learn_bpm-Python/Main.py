import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Net import Net

def main():
    dataset = pd.read_csv("winequality-red.csv", sep=';', decimal=",", dtype=np.float)
    x=dataset.iloc[:,:-1].values
    y=dataset.iloc[:,-1:].values
    y = OneHotEncoder().fit_transform(y).toarray().astype(float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    epoch = 1
    topology = [len(x_train[0]), len(x_train[0]), len(x_train[0]), len(y_train[0])]
    my_net = Net(topology)

    for e in range(epoch):
        for r in range(len(x_train)):
            input_vals = x_train[r]
            target_vals = y_train[r]
            my_net.feedForward(input_vals)
            #my_net.backProp(target_vals)
            print("Data row: {}".format(r))
            my_net.getResults()

    print()

main()

