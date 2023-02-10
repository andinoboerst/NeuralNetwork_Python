import numpy as np
import matplotlib.pyplot as plt
import myPackages.neural_network as nn




def main():
    print("Start neural netwrok.")
    my_NN = nn.NeuralNetwork()
    my_NN.add_layer(3, layer_type="input")
    my_NN.add_layer(5)
    my_NN.add_layer(3)
    my_NN.add_layer(1, "output")

    #print([node.bias for node in my_NN.layers[0].nodes])
    print(my_NN.predict([1, 2, 0.3]))


if __name__=="__main__":
    main()