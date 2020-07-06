from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from random import seed
from random import random


# gets the average of all values inside the array.  Used for testing the function's approximations
def array_average(in_array=[]):

    sum = 0
    for x in range(len(in_array)):
        sum += in_array[x]

    return sum / len(in_array)


seed(6)

X, y = make_regression(n_samples=3, random_state=1)

# print(X)
# print(y)

inputArrays = []
answerArrays = []

# will build a new neural network based on the provided parameters.
clf = MLPRegressor(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(25, 10, 5, 3), random_state=4, activation='identity',
                   learning_rate='adaptive')
clf.n_outputs_ = 1

# this for loop will build a sample data set for us, in a real life scenario
# we would likely read the data set in from a file
for x in range(1000):
    # This will be the input data set
    inarray1 = [random() * 500, random() * 500, random() * 500, random() * 500, random() * 500, random() * 500,
                random() * 500]
    # inArray2 will contain the labels for the datsets, this will have no impact on the data analysis results
    inarray2 = [0, 1, 2, 3, 4, 5, 6]

    # The answer, as a single float, will be the output data set.
    # This value will be used to back-propagate the neural network
    answerOutput = array_average(inarray1)

    array2d = (inarray1, inarray2)
    inputArrays.append(inarray1)
    answerArrays.append(answerOutput)

# The 0.01 represents the fraction of the dataset that will be used as test sample.  The function will automatically
# sort the data into test and train samples
X_train, X_test, y_train, y_test = train_test_split(inputArrays, answerArrays, random_state=1, test_size=0.01)

# print(X_train)

# This function will actually start training the algorithm, based on the training data provided by the user
clf.fit(X_train, y_train)

# This function will store all of the test results in an array.
results = clf.predict(X_test)

# This output function will display a comparison between the function's approximation, and the user's known answers
print("RESULTS: ")
for x in range(len(results)):
    print("PREDICTION", x, ":", results[x])
    print("ANSWER", x, ":", y_test[x])
