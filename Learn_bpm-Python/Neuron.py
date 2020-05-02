from random import seed
from random import random
from numpy import tanh


class Connection:
    def __init__(self):
        self.weight = 0
        self.delta_weight = 0


class Neuron:
    def __init__(self, num_outputs, my_index):
        seed(1)
        self.__m_my_index = my_index
        self.__m_output_value = 0
        # list of connections
        self.__m_output_weights = []
        for c in range(num_outputs):
            self.__m_output_weights.append(Connection())
            self.__m_output_weights[c].weight = self.randomWeight()

    @staticmethod
    def randomWeight():
        return random()

    def transferFunction(self, x):
        return tanh(x)

    def transferFunctionDerivative(self, x):
        return 1.0 - x * x

    def getOutputValue(self):
        return self.__m_output_value

    def setOutputValue(self, value):
        self.__m_output_value = value

    def feedForward(self, prev_layer):
        summ = 0.0
        for n in range(len(prev_layer)):
            summ += prev_layer[n].getOutputValue() * prev_layer[n].__m_output_weights[self.__m_my_index].weight
        self.__m_output_value = self.transferFunction(summ)
