from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from random import seed
from random import random


def arrayaverager(array1=[], array2=[]):
    returnarray = []

    for x in range(len(array1)):
        appendValue = (array1[x] + array2[x]) / 2
        returnarray.append(appendValue)
    return returnarray


# gets the average of all values inside the array.  Used for testing the function's approximations
def arrayAverage(inArray=[]):

    sum = 0
    for x in range(len(inArray)):
        sum += inArray[x]

    return sum / len(inArray)


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

# this for loop will build a sample data set for us, in a real life scenario,
# we would likely read the data set in from a file
for x in range(1000):
    # This will be the input data set
    inarray1 = [random() * 500, random() * 500, random() * 500, random() * 500, random() * 500, random() * 500,
                random() * 500]
    # inArray2 will contain the labels for the datsets, this will have no impact on the data analysis results
    inarray2 = [0, 1, 2, 3, 4, 5, 6]

    # newOutput = arrayaverager(inarray1, inarray2)
    # The answer, as a single float, will be the output data set.
    # This value will be used to back-propagate the neural network
    answerOutput = arrayAverage(inarray1)

    array2d = (inarray1, inarray2)
    inputArrays.append(inarray1)
    answerArrays.append(answerOutput)

# The .0001 represents the fraction of the dataset that will be used as test sample.  The function will automatically
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


# print(clf.score(X_test, y_test))



# trains the algorithm
# for x in range(10000):
#    if x % 500 == 0:
#        print(x)
    # print(inputArrays[x])
    # print(answerArrays[x])
#    clf.fit(inputArrays[x], answerArrays[x])


# for x in range(20):
#    inarray1 = [random(), random()]
#    inarray2 = [0., 1.]

#    array2d = (inarray1, inarray2)

    # answer = arrayaverager(inarray1, inarray2)
#    answer = arrayModifier(inarray1)

#    output = clf.predict(array2d)

#    print("Answer: ")
#    print(answer)
#    print("Prediction: ")
#    print(output)
