from Neuron import Neuron
import math

class Net:
    def __init__(self, topology):
        self.__m_error = 0
        self.__m_recent_average_error = 0
        self.__m_recent_average_smoothing_factor = 0
        self.__m_layers = []
        num_layers = len(topology)
        # Utworzenie warstw i neuronow
        for layer_num in range(num_layers):
            self.__m_layers.append([])
            num_outputs = 0
            if layer_num != (num_layers - 1):
                num_outputs = topology[layer_num + 1]
            for neuron_num in range(topology[layer_num] + 1):
                self.__m_layers[layer_num].append(Neuron(num_outputs, neuron_num))
            # Dodanie biasu do ostatnie neuronu kazdej warstwy
            self.__m_layers[-1][-1].setOutputValue(1.0)

    def feedForward(self, input_values):
        # Sprawdzenie czy zgadza sie ilosc danych wejsciowych (bez bias neuronu)
        assert len(input_values) == (len(self.__m_layers[0]) - 1), "Wrong input values"

        # Wypelnienie neuronow pierwszej warstwy wartosciami (bez bias)
        for i in range(len(input_values)):
            self.__m_layers[0][i].setOutputValue(input_values[i])

        # Propagacja do przodu - feed forward (bez bias neuronu)
        for layer_num in range(1, len(self.__m_layers)):
            prev_layer = self.__m_layers[layer_num - 1]
            for n in range(len(self.__m_layers[layer_num]) - 1):
                self.__m_layers[layer_num][n].feedForward(prev_layer)

    def backProp(self, target_values):
        # Obliczanie calkowitego bledu (RMS) (bez bias)
        output_layer = self.__m_layers[-1]
        self.__m_error = 0.0
        for n in range(len(output_layer) - 1):
            delta = target_values[n] - output_layer[n].getOutputValue()
            self.__m_error += math.pow(delta, 2)
        self.__m_error /= (len(output_layer) - 1)
        self.__m_error = math.sqrt(self.__m_error) # RMS

        # Implementacja ostatniej sredniej z pomiarow
        self.__m_recent_average_error = (self.__m_recent_average_error * self.__m_recent_average_smoothing_factor + self.__m_error) / (self.__m_recent_average_smoothing_factor + 1)

        # Obliczanie gradientu warstwy wyjsciowej (bez bias)
        for n in range(len(output_layer) - 1):
            output_layer[n].calcOutputGradients(target_values[n])

        # Obliczanie gradientu warstwy ukrytej (od przedostatniej warstwy do 2 warstwy naszej sieci) (z bias)
        for layer_num in range(len(self.__m_layers) - 2, 0, -1):
            hidden_layer = self.__m_layers[layer_num]
            next_layer = self.__m_layers[layer_num + 1]
            for n in range(len(hidden_layer)):
                hidden_layer[n].calcHiddenGradients(next_layer)

        # Aktualizowanie wag polaczen dla wszystkich warstw od warstwy wyjsciowej do ostatniej ukrytej warstwy (z bias)
        for layer_num in range(len(self.__m_layers) - 1, 0, -1):
            layer = self.__m_layers[layer_num]
            prev_layer = self.__m_layers[layer_num - 1]
            for n in range(len(layer) - 1):
                layer[n].updateInputWeights(prev_layer)

    def getResults(self):
        # Zwrocenie otrzymanego outputu (bez neuronow z bias)

        result_values = []
        for i in range(len(self.__m_layers[-1]) - 1):
            output_value = self.__m_layers[-1][i].getOutputValue()
            result_values.append(output_value)
        return result_values
