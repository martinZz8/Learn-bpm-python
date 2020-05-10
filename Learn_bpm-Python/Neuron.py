from random import uniform
from numpy import exp


class Connection:
    def __init__(self):
        self.weight = 0
        self.delta_weight = 0


class Neuron:

    # Statyczne wartosci uzywane we wstecznej propagacji
    eta = 0.002
    alpha = 0.9

    def __init__(self, num_outputs, my_index):
        self.__m_my_index = my_index
        self.__m_output_value = 0
        self.__m_gradient = 0

        # Lista polaczen
        self.__m_output_weights = []
        # Ustawianie wag polaczen (rowniez dla bias)
        for c in range(num_outputs):
            self.__m_output_weights.append(Connection())
            self.__m_output_weights[c].weight = self.randomWeight()

    @staticmethod
    def randomWeight():
        return uniform(0, 1)

    @staticmethod
    def transferFunction(x):
        # Funkcja sigmoid
        return 1 / (1 + exp(-x))

    @staticmethod
    def transferFunctionDerivative(x):
        # Pochodna funkcji sigmoid
        p = exp(-x)
        return p / (1 + p)**2

    def getOutputValue(self):
        return self.__m_output_value

    def setOutputValue(self, value):
        self.__m_output_value = value

    def feedForward(self, prev_layer):
        summ = 0.0
        for n in range(len(prev_layer)):
            summ += prev_layer[n].getOutputValue() * prev_layer[n].__m_output_weights[self.__m_my_index].weight
        self.__m_output_value = self.transferFunction(summ)

    def calcOutputGradients(self, target_value):
        delta = target_value - self.__m_output_value
        self.__m_gradient = delta * self.transferFunctionDerivative(self.__m_output_value)

    def sumDOW(self, next_layer):
        summ = 0.0

        # Sumowanie naszych wkladow z errorow w wezlach, ktore nakarmilismy (bez bias, bo jego nie karmilismy)
        for n in range(len(next_layer) - 1):
            summ += self.__m_output_weights[n].weight * next_layer[n].__m_gradient
        return summ

    def calcHiddenGradients(self, next_layer):
        dow = self.sumDOW(next_layer)
        self.__m_gradient = dow * self.transferFunctionDerivative(self.__m_output_value)

    def updateInputWeights(self, prev_layer):

        # Wagi, ktore mamy zaktualizowac, znajduja sie w liscie polaczen w neuronach z poprzedniej warstwy (aktualizujemy tez wage bias)
        for n in range(len(prev_layer)):
            neuron = prev_layer[n]
            old_delta_weight = neuron.__m_output_weights[self.__m_my_index].delta_weight

            # Indywidualne wejscie, zalezne od gradientu i wspolczynnika nauczania
            new_delta_weight = Neuron.eta * neuron.getOutputValue() * self.__m_gradient + Neuron.alpha * old_delta_weight
            neuron.__m_output_weights[self.__m_my_index].delta_weight = new_delta_weight
            neuron.__m_output_weights[self.__m_my_index].weight += new_delta_weight
            # print("%.4f" % neuron.__m_output_weights[self.__m_my_index].weight)
