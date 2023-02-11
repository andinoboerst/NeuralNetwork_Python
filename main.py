import numpy as np
import matplotlib.pyplot as plt
import myPackages.neural_network as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd


def main():
    print("Start neural netwrok.")

    digits = datasets.load_digits()
    # plot_digits(digits)

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))/100

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    y_train_adjusted = np.zeros((len(y_test), 10)) # for every digit store an array of length 10 of zeros with the correct digit being represented with a 1 in that index
    for index, digit in enumerate(y_train):
        y_train_adjusted[index, digit] = 1


    my_NN = nn.NeuralNetwork()
    my_NN.add_input_layer(64)
    #my_NN.add_standard_layer(53)
    my_NN.add_standard_layer(42)
    my_NN.add_standard_layer(31)
    #my_NN.add_standard_layer(20)
    my_NN.add_output_layer(10)

    my_NN.fit(X_train, y_train_adjusted)

    #print([node.bias for node in my_NN.layers[0].nodes])
    #print(my_NN.predict(np.array([[1, 2, 0.3],[1,2,0.9]])))
    predictions_test = my_NN.predict(X_test)
    num_predictions_test = np.argmax(predictions_test, axis=1)
    print(predictions_test)
    for pred, actual in zip(num_predictions_test, y_test):
        print(f"pred: {pred} | {actual} :actual")
    print(num_predictions_test)
    print(f"accuracy: {sum(num_predictions_test==y_test)/len(y_test)}")

    df = pd.DataFrame(0, index=np.arange(10), columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    for pre, actual in zip(num_predictions_test, y_test):
        df.loc[pre,actual] += 1
    sns.heatmap(df, annot=True)
    plt.ylabel("predicted")
    plt.xlabel("actual")
    plt.show()


def plot_digits(digits):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    plt.show()

if __name__=="__main__":
    main()