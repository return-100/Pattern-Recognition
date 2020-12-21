import numpy as np
import math as math


class Features:
    def __init__(self):
        self.number_of_classes = None
        self.x = None
        self.class_no = []
        self.num_of_features = None
        self.total_num_of_sample = None

    def read_feature_value(self, filename):
        file = open(filename, "r")
        line = file.readline().split()
        self.num_of_features = int(line[0])
        self.number_of_classes = int(line[1])
        self.total_num_of_sample = int(line[2])
        self.x = np.zeros((self.total_num_of_sample, self.num_of_features + 1))
        idx = 0
        while line:
            line = file.readline().split()
            if len(line) == 0:
                break
            for i in range(self.num_of_features):
                self.x[idx][i] = line[i]
            self.x[idx][self.num_of_features] = 1
            self.class_no.append(int(line[self.num_of_features]))
            idx = idx + 1
        file.close()


class Perceptron:
    def __init__(self, features):
        self.features = features
        self.rho = 0.3
        self.w = None
        self.iteration = 500

    def predict_class(self, w, x):
        dot_product = np.dot(w, x.T)
        if dot_product >= 0.0:
            return 1
        else:
            return 2

    def classify(self, w, i):
        predicted_class = self.predict_class(w, self.features.x[i])
        if predicted_class == 1 and self.features.class_no[i] == 2:
            return False
        elif predicted_class == 2 and self.features.class_no[i] == 1:
            return False
        else:
            return True

    def grad_descent_val(self, y_set):
        dw = np.zeros((1, self.features.num_of_features + 1))
        for i in range(len(y_set)):
            if self.features.class_no[y_set[i]] == 1:
                dw += (-1 * self.features.x[y_set[i]])
            else:
                dw += self.features.x[y_set[i]]
        return dw

    def train(self):
        self.w = np.zeros((self.iteration + 1, self.features.num_of_features + 1))
        self.w[0] = np.random.random((1, self.features.num_of_features + 1))
        for t in range(self.iteration):
            y_set = []
            for i in range(self.features.total_num_of_sample):
                if not self.classify(self.w[t], i):
                    y_set.append(i)
            # print(len(y_set), " ", i)
            # self.grad_descent_val(y_set)
            self.w[t + 1] = self.w[t] - self.rho * self.grad_descent_val(y_set)

    def test(self, filename):
        file = open(filename, "r")
        line = file.readline().split()
        x = np.zeros((1, self.features.num_of_features + 1))
        x[0][self.features.num_of_features] = 1
        total_sample = 0
        correct_classification = 0
        while line:
            if len(line) == 0:
                break
            for i in range(self.features.num_of_features):
                x[0][i] = line[i]
            predicted_class = self.predict_class(self.w[self.iteration - 1], x)
            if predicted_class == int(line[self.features.num_of_features]):
                correct_classification += 1
            else:
                print(total_sample + 1, x, line[self.features.num_of_features], predicted_class)
            total_sample += 1
            line = file.readline().split()
        file.close()
        print("Accuracy:", (correct_classification / total_sample) * 100, "%\n")


class RewardAndPunishment:
    def __init__(self, features):
        self.features = features
        self.rho = 0.3
        self.w = None
        self.iteration = 1000

    def predict_class(self, w, x):
        dot_product = np.dot(w, x.T)
        if dot_product > 0.0:
            return 1
        else:
            return 2

    def punishment(self, predicted_class, t):
        if predicted_class == 1:
            self.w[t + 1] = self.w[t] - self.rho * self.features.x[t]
        else:
            self.w[t + 1] = self.w[t] + self.rho * self.features.x[t]

    def train(self):
        self.w = np.zeros((self.features.total_num_of_sample + 1, self.features.num_of_features + 1))
        for i in range(self.iteration):
            correct_classifcation = 0
            for t in range(self.features.total_num_of_sample):
                predicted_class = self.predict_class(self.w[t], self.features.x[t])
                if predicted_class == self.features.class_no[t]:
                    self.w[t + 1] = self.w[t]
                    correct_classifcation += 1
                else:
                    self.punishment(predicted_class, t)
            self.w[0] = self.w[self.features.total_num_of_sample]
            if correct_classifcation == self.features.total_num_of_sample:
                # print(i + 1)
                break

    def test(self, filename):
        file = open(filename, "r")
        line = file.readline().split()
        x = np.zeros((1, self.features.num_of_features + 1))
        x[0][self.features.num_of_features] = 1
        total_sample = 0
        correct_classification = 0
        while line:
            if len(line) == 0:
                break
            for i in range(self.features.num_of_features):
                x[0][i] = line[i]
            predicted_class = self.predict_class(self.w[self.features.total_num_of_sample], x)
            if predicted_class == int(line[self.features.num_of_features]):
                correct_classification += 1
            else:
                print(total_sample + 1, x, line[self.features.num_of_features], predicted_class)
            total_sample += 1
            line = file.readline().split()
        file.close()
        print("Accuracy:", (correct_classification/total_sample) * 100, "%\n")


class Pocket:
    def __init__(self, features):
        self.features = features
        self.rho = 0.3
        self.w = None
        self.ws = None
        self.hs = 0
        self.iteration = 500

    def predict_class(self, w, x):
        dot_product = np.dot(w, x.T)
        if dot_product >= 0.0:
            return 1
        else:
            return 2

    def classify(self, w, i):
        predicted_class = self.predict_class(w, self.features.x[i])
        if predicted_class != self.features.class_no[i]:
            return False
        else:
            return True

    def grad_descent_val(self, y_set):
        dw = np.zeros((1, self.features.num_of_features + 1))
        for i in range(len(y_set)):
            if self.features.class_no[y_set[i]] == 1:
                dw += (-1 * self.features.x[y_set[i]])
            else:
                dw += self.features.x[y_set[i]]
        return dw

    def train(self):
        self.w = np.zeros((self.iteration + 1, self.features.num_of_features + 1))
        self.w[0] = np.random.random((1, self.features.num_of_features + 1))
        self.ws = np.zeros((1, self.features.num_of_features + 1))
        for t in range(self.iteration):
            y_set = []
            h = 0
            for i in range(self.features.total_num_of_sample):
                if not self.classify(self.w[t], i):
                    y_set.append(i)
            self.w[t + 1] = self.w[t] - self.rho * self.grad_descent_val(y_set)
            for i in range(self.features.total_num_of_sample):
                if self.classify(self.w[t + 1], i):
                    h += 1
            if h > self.hs:
                self.hs = h
                self.ws = self.w[t + 1]

    def test(self, filename):
        file = open(filename, "r")
        line = file.readline().split()
        x = np.zeros((1, self.features.num_of_features + 1))
        x[0][self.features.num_of_features] = 1
        total_sample = 0
        correct_classification = 0
        while line:
            if len(line) == 0:
                break
            for i in range(self.features.num_of_features):
                x[0][i] = line[i]
            predicted_class = self.predict_class(self.ws, x)
            if predicted_class == int(line[self.features.num_of_features]):
                correct_classification += 1
            else:
                print(total_sample + 1, x, line[self.features.num_of_features], predicted_class)
            total_sample += 1
            line = file.readline().split()
        file.close()
        print("Accuracy:", (correct_classification / total_sample) * 100, "%")


if __name__ == "__main__":
    linear_feature_class = Features()
    linear_feature_class.read_feature_value("trainLinearlySeparable.txt")

    non_linear_feature_class = Features()
    non_linear_feature_class.read_feature_value("trainLinearlyNonSeparable.txt")

    print("Result of Perceptron algorithm")
    perceptron = Perceptron(linear_feature_class)
    perceptron.train()
    perceptron.test("testLinearlySeparable.txt")

    print("Result of RP algorithm")
    rnp = RewardAndPunishment(linear_feature_class)
    rnp.train()
    rnp.test("testLinearlySeparable.txt")

    print("Result of Pocket algorithm")
    pocket = Pocket(non_linear_feature_class)
    pocket.train()
    pocket.test("testLinearlyNonSeparable.txt")
