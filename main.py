import numpy as np
import matplotlib.pyplot as plt
import myPackages.neural_network as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split


def main():
    print("Start neural netwrok.")

    digits = datasets.load_digits()
    # plot_digits(digits)

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    my_NN = nn.NeuralNetwork()
    my_NN.add_layer(3, layer_type="input")
    my_NN.add_layer(5)
    my_NN.add_layer(3)
    my_NN.add_layer(1, "output")

    #print([node.bias for node in my_NN.layers[0].nodes])
    print(my_NN.predict([1, 2, 0.3]))


def plot_digits(digits):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    plt.show()

if __name__=="__main__":
    main()