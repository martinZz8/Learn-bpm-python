import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import Net

def main():
    dataset = pd.read_csv("winequality-red.csv", sep=';', decimal=",", dtype=np.float)
    x=dataset.iloc[:,:-1].values
    y=dataset.iloc[:,-1:].values
    y = OneHotEncoder().fit_transform(y).toarray().astype(float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    bias = 1.0
    epoch = 1

    for e in range(epoch):
        for r in range(len(x_train)):
            input_vals = x_train[r]
            target_vals = y_train[r]
            topology = [len(x_train[r]), len(x_train[r]), len(x_train[r]), len(y_train[r])]
            my_net = Net(topology)
            my_net.feedForward(input_vals)
            #my_net.backProp(target_vals)
            #my_net.getResults(result_vals)

    print()

main()

