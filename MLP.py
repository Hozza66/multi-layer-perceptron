# Multi-Layer ANN from Scratch
# Authur: Haoran Hong

# Imports
import numpy as np
import numpy.random
import pandas as pd
import time

# Read dataset
data = pd.read_csv("data.csv")

# Shuffle dataset and perform train/test split, ratio set here
data = data.sample(frac=1)
train_size = int(0.7 * len(data))
train = data[:train_size]
test = data[train_size:]

# Sort data set into x and y for input and class
X_train = train.drop(columns=['Column5'])
y_train = train['Column5']
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

X_test = test.drop(columns=['Column5'])
y_test = test['Column5']
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()


# Create Multi-layer Perceptron class
class MLP:

    # Definitions for activation functions, derivative, and loss functions
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derive(selfself, a):
        return a * (1 - a)

    def tanh(self, z):
        return np.tanh(z)

    def tanh_derive(self, a):
        return 1 - np.tanh(a) * np.tanh(a)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derive(self, a):
        return 1. * (a > 0)

    def softmax(self, z):
        expo = np.exp(z)
        expo_sum = np.sum(np.exp(z))
        return expo / expo_sum

    def softmax_derive(selfself, a):
        return a * (1 - a)

    def copy(self, a):
        return a + 0

    def cross_en_loss(self, y_hat, y):
        return y_hat - y

    # Initiate mlp structures
    # no_hidden is the parameter for hidden layers and neutrons within
    # Number of hidden layers and neurons in each layer can be set to any number
    def __init__(self, no_input=4, no_hidden=[8], no_output=1):

        # Variables
        self.total_time = time.perf_counter()
        self.no_input = no_input
        self.no_hidden = no_hidden
        self.no_output = no_output
        self.weight_array = []
        self.bias_array = []
        self.activation_array = []
        self.derive_array = []
        self.delta_array = []
        self.total_correct_count = 0
        self.total_incorrect_count = 0
        self.correct_count = 0
        self.incorrect_count = 0
        self.instance_count = 0
        self.current_accuracy = 0
        self.epoch_accuracy = 0
        self.accuracy_list = []
        self.prediction = 0
        self.actual = 0
        self.epoch = 1
        self.total_instance_count = 0
        self.hidden_layer_function = ''
        self.output_layer_function = ''
        self.epoch_time = time.perf_counter()
        self.data_set = 3

        self.learning_rate = 0.6

        # Shows mlp structure defined above
        self.layers = [no_input] + no_hidden + [no_output]

        # Create a array of matrices to the shape of the mlp to store weights
        # Random values are used before training
        for i in range(len(self.layers) - 1):
            weight = np.random.rand(self.layers[i + 1], self.layers[i])
            self.weight_array.append(weight)

        # Create a array of matrices to the shape of the mlp to store basis
        # Random values are used training
        for i in range(len(self.layers) - 1):
            bias = np.random.rand(self.layers[i + 1], 1)
            self.bias_array.append(bias)

        # Create a list of matrices to the shape of the mlp for to store delta array
        # Random values are used training
        for i in range(len(self.layers) - 1):
            delta = np.random.rand(self.layers[i + 1], 1)
            self.delta_array.append(np.matrix(delta))

    # Function for forward propagation
    def forward_prop(self, inputs):

        self.epoch_time = time.perf_counter()

        # Append the inputs as "activation" layer
        activation_output = np.matrix(inputs)
        self.activation_array.append(activation_output)

        # Counts for loops
        bias_layer_count = 0
        current_layer = 0

        # For each layer of weights
        for weight in self.weight_array:

            # Dot product of current layer weights and previous layer activations
            dot_product = np.dot(weight, activation_output.T)

            # Add bias
            z = dot_product + self.bias_array[bias_layer_count]
            z = z.T

            # Set the activation function for all hidden layers
            # Append derivative function
            if current_layer != (len(self.layers) - 2):
                self.hidden_layer_function = 'Relu'
                activation_output = self.relu(z)
                self.derive_array.append(np.matrix(self.relu_derive(np.array(activation_output))))

            # Set the activation function for the output layer
            # Due to 1 or 0 classification output layer activation function is always sigmoid
            # Append derivative function
            if current_layer == (len(self.layers) - 2):
                self.output_layer_function = 'sigmoid'
                activation_output = self.sigmoid(z)
                self.derive_array.append(np.matrix(self.sigmoid_derive(np.array(activation_output))))

            # Append activation
            self.activation_array.append(activation_output)

            bias_layer_count += 1
            current_layer += 1

        return activation_output

    # Function for back propagation for training set
    def back_prop(self, prediction):

        # Assign predicted and actual output value
        self.prediction = prediction
        self.actual = y_train[self.instance_count]

        # Counts for loops
        reverse_count = len(self.layers) - 2
        count1 = 0
        count2 = 0
        count3 = 0
        reset = 0

        # Error calculation
        error = self.cross_en_loss(self.prediction, self.actual)

        # Delta calculation for output layer
        self.delta_array[reverse_count] = error * self.derive_array[reverse_count]

        # Reverse iteration through delta array
        for i in reversed(self.delta_array):

            if count1 == 0:
                pass

            if count1 != 0:
                # Iterate through each delta value of a layer
                for j in i:
                    # Dot product of next layer delta and current layer weight values
                    dot_product = np.dot(self.delta_array[reverse_count + 1].T,
                                         self.weight_array[reverse_count + 1][:, count3])
                    # Calculate and update current delta value
                    self.delta_array[reverse_count][count3] = \
                        self.derive_array[reverse_count][count2, count3] * dot_product
                    count3 += 1

            count3 = reset
            count1 += 1
            reverse_count -= 1

        count1 = 0

        # Iterate trough each layer of weights
        for i in self.weight_array:
            # Calculate and update weights using current layer delta and previous layer activation values
            self.weight_array[count1] = \
                self.weight_array[count1] - \
                (self.learning_rate * np.dot(self.delta_array[count1], self.activation_array[count1]))
            count1 += 1
        count1 = 0
        count3 = 0
        reset = 0
        bias_con = 1

        # Iterate trough each layer of bais
        for i in self.bias_array:
            # Iterate through each bias of a layer
            for j in i:
                # Calculating and updating bais values using current layer delta values
                self.bias_array[count1][count3] = (self.bias_array[count1][count3] - (
                        self.learning_rate * self.delta_array[count1][count3] * bias_con))
                count3 += 1
            count3 = reset
            count1 += 1

        # Class predicted values to 0 or 1
        if self.prediction > 0.5:
            self.prediction = 1

        if self.prediction < 0.5:
            self.prediction = 0

        # Count if prediction is correct or incorrect
        if self.prediction == self.actual:
            self.correct_count += 1
            self.total_correct_count += 1

        if self.prediction != self.actual:
            self.incorrect_count += 1
            self.total_incorrect_count += 1

        # Calculate accuracies
        self.current_accuracy = self.total_correct_count / (self.total_instance_count + 1)
        self.epoch_accuracy = self.correct_count / (self.instance_count + 1)
        self.accuracy_list.append(self.epoch_accuracy)

        # Reset activation and derivative arrays for next instance of input
        self.activation_array = []
        self.derive_array = []
        self.instance_count += 1
        self.total_instance_count += 1


    # Function for rounding to 3dp
    def r3(self, x):
        return "{:.3f}".format(round(x, 3))

    # Function for rounding to 5dp
    def r5(self, x):
        return "{:.5f}".format(round(x, 5))

    # Function to display results for each epoch
    def results(self):
        print('Epoch :', self.epoch)
        print('ANN Layer Structure: ', self.layers)
        print('Hidden layers function: ', self.hidden_layer_function)
        print('Output function: ', self.output_layer_function)
        print('Learning Rate: ', self.learning_rate)
        print('Epoch Correct counts: ', self.correct_count)
        print('Epoch Incorrect counts: ', self.incorrect_count)
        print('Total instances: ', self.total_instance_count)
        print('Total Correct counts: ', self.total_correct_count)
        print('Total Incorrect counts: ', self.total_incorrect_count)
        print('Epoch Run Time: ', self.r5(time.perf_counter() - self.epoch_time))
        print('Total Run Time: ', self.r3(time.perf_counter() - self.total_time))
        print('Epoch Accuracy: ', self.r3(self.epoch_accuracy*100))
        print('Current Accuracy: ', self.r3(self.current_accuracy))
        print('Mean Accuracy', self.r3(np.mean(self.accuracy_list)), '±', self.r3(np.std(self.accuracy_list)), '\n\n')

        self.instance_count = 0
        self.correct_count = 0
        self.incorrect_count = 0
        self.epoch += 1

    # Function to display results for testing
    def results_test(self):
        print('ANN Layer Structure: ', self.layers)
        print('Hidden layers function: ', self.hidden_layer_function)
        print('Output function: ', self.output_layer_function)
        print('Learning Rate: ', self.learning_rate)
        print('Correct counts: ', self.correct_count)
        print('Incorrect counts: ', self.incorrect_count)
        print('Instances: ', self.total_instance_count)
        print('Total Run Time: ', self.r3(time.perf_counter() - self.total_time))
        print('Accuracy: ', self.r3(self.epoch_accuracy*100), '\n\n')

        self.instance_count = 0
        self.correct_count = 0
        self.incorrect_count = 0
        self.epoch += 1

    # Function to reset variables for testing
    def reset(self):
        self.activation_array = []
        self.derive_array = []
        self.delta_array = []
        self.correct_count = 0
        self.incorrect_count = 0
        self.instance_count = 0
        self.current_accuracy = 0
        self.epoch_accuracy = 0
        self.accuracy_list = []
        self.prediction = 0
        self.actual = 0
        self.epoch = 1
        self.total_instance_count = 0
        self.epoch_time = time.perf_counter()

    # Function for back propagation for testing set
    # Same as the function for training minus weight and bias update
    def back_prop_test(self, prediction):
        self.prediction = prediction
        self.actual = y_test[self.instance_count]
        if self.prediction > 0.5:
            self.prediction = 1
        if self.prediction < 0.5:
            self.prediction = 0
        if self.prediction == self.actual:
            self.correct_count += 1
            self.total_correct_count += 1
        if self.prediction != self.actual:
            self.incorrect_count += 1
            self.total_incorrect_count += 1
        self.current_accuracy = self.total_correct_count / (self.total_instance_count + 1)
        self.epoch_accuracy = self.correct_count / (self.instance_count + 1)
        self.accuracy_list.append(self.epoch_accuracy)
        self.activation_array = []
        self.derive_array = []
        self.instance_count += 1
        self.total_instance_count += 1

    # Function to output final accuracy and total run time
    def accuracy_time(self):
        print(self.epoch_accuracy)
        print(self.r3(time.perf_counter() - self.total_time))

    # Function for results to output to file
    def output_to_file_train(self):
        output_train = "Training:\n"
        output_train += 'Epoch:' + str(self.epoch) +"\n"
        output_train +='ANN Layer Structure: ' + str(self.layers) +"\n"
        output_train +='Hidden layers function: ' + str(self.hidden_layer_function) +"\n"
        output_train +='Output function: ' + str(self.output_layer_function) +"\n"
        output_train +='Learning Rate: ' + str(self.learning_rate) +"\n"
        output_train +='Epoch Correct counts: ' + str(self.correct_count) +"\n"
        output_train +='Epoch Incorrect counts: ' + str(self.incorrect_count) +"\n"
        output_train +='Total instances: ' + str(self.total_instance_count) +"\n"
        output_train +='Total Correct counts: ' + str(self.total_correct_count) +"\n"
        output_train +='Total Incorrect counts: ' + str(self.total_incorrect_count) +"\n"
        output_train +='Epoch Run Time: ' + str(self.r5(time.perf_counter() - self.epoch_time)) +"\n"
        output_train +='Total Run Time: ' + str(self.r3(time.perf_counter() - self.total_time)) +"\n"
        output_train +='Epoch Accuracy: ' + str(self.r3(self.epoch_accuracy)) +"\n"
        output_train +='Current Accuracy: ' + str(self.r3(self.current_accuracy)) +"\n"
        output_train +='Mean Accuracy' + str(self.r3(np.mean(self.accuracy_list)) + '±' + self.r3(np.std(self.accuracy_list))) + '\n\n'

        return output_train

    # Function for results to output to file
    def output_to_file_test(self):
        output_test = "Testing:\n"
        output_test += 'ANN Layer Structure: ' + str(self.layers) +"\n"
        output_test +='Hidden layers function: ' + str(self.hidden_layer_function) +"\n"
        output_test +='Output function: ' + str(self.output_layer_function) +"\n"
        output_test +='Learning Rate: ' + str(self.learning_rate) +"\n"
        output_test +='Correct counts: ' + str(self.correct_count) +"\n"
        output_test +='Incorrect counts: ' + str(self.incorrect_count) +"\n"
        output_test +='Instances: ' + str(self.total_instance_count) +"\n"
        output_test +='Total Run Time: ' + str(self.r3(time.perf_counter() - self.total_time)) +"\n"
        output_test +='Accuracy: ' + str(self.r3(self.epoch_accuracy)) + '\n\n'

        return output_test
# Main
if __name__ == '__main__':
    #Show size of training and test sets
    print('Train dataset size: ', len(X_train))
    print('Test dataset size: ', len(X_test))
    output = 'Train dataset size: ' + str(len(X_train)) + "\n"
    output +='Test dataset size: ' + str(len(X_test))

    # Create MLP object
    mlp = MLP()



    # For loop for training with training set, epoch can be set here in range
    for epoch in range(10):
        print("Training:")
        # Iterate through each instance and perform SGD
        for i in range(len(X_train)):
            output1 = mlp.forward_prop(X_train[i])
            mlp.back_prop(output1)
        output += mlp.output_to_file_train()
        mlp.results()


    mlp.reset()

    # Run through testing set after training
    print("Testing:")
    for i in range(len(X_test)):
        output2 = mlp.forward_prop(X_test[i])
        mlp.back_prop_test(output2)
    output += mlp.output_to_file_test()
    mlp.results_test()

    #Write test results to text file
    with open('test_results.txt', 'w') as f:
        f.write(output)
