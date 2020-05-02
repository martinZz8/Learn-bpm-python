from Neuron import Neuron


class Net:
    def __init__(self, topology):
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

    def feedForward(self, input_vals):
        # Sprawdzenie czy zgadza sie ilosc danych wejsciowych (bez bias neuronu)
        assert len(input_vals) == (len(self.__m_layers[0]) - 1), "Wrong input vals"

        # Wypelnienie neuronow pierwszej warstwy wartosciami (bez bias)
        for i in range(len(input_vals)):
            self.__m_layers[0][i].setOutputValue(input_vals[i])

        # Propagacja do przodu (bez bias neuronu)
        for layer_num in range(1, len(self.__m_layers)):
            prev_layer = self.__m_layers[layer_num - 1]
            for n in range(len(self.__m_layers[layer_num]) - 1):
                self.__m_layers[layer_num][n].feedForward(prev_layer)

    # def backProp(self, target_vals):

    def getResults(self):
        file = open("Results.txt", "a")

        for i in range(len(self.__m_layers[-1])):
            output_value = self.__m_layers[-1][i].getOutputValue()
            if i != (len(self.__m_layers[-1]) - 1):
                print("%.5f" % output_value, end=", ")
                file.write("%.5f, " % output_value)
            else:
                print("%.5f" % output_value)
                file.write("%.5f" % output_value)
        print()
        file.write("\n\n")
        file.close()
