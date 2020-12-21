import numpy as np
import copy


class Features:
    def __init__(self):
        self.number_of_classes = 0
        self.x = []
        self.y = []
        self.class_set = set()
        self.class_no = []
        self.num_of_features = None
        self.num_of_sample = 0

    def count_num_of_sample(self, filename):
        file = open(filename, "r")
        line = file.readline().split()
        self.num_of_features = int(len(line)) - 1
        while len(line) != 0:
            self.num_of_sample += 1
            self.class_set.add(int(line[self.num_of_features]))
            line = file.readline().split()
        self.number_of_classes = len(self.class_set)
        file.close()

    def read_feature_value(self, filename):
        file = open(filename, "r")
        for i in range(self.num_of_sample):
            x = np.zeros((self.num_of_features, 1))
            y = np.zeros((self.number_of_classes, 1))
            line = file.readline().split()
            for j in range(self.num_of_features):
                x[j][0] = float(line[j])
            y[int(line[self.num_of_features]) - 1][0] = 1.0
            self.x.append(x)
            self.y.append(y)
            self.class_no.append(int(line[self.num_of_features]))
        file.close()

    def normalize_input(self):
        self.x = np.array(self.x)
        self.x = (self.x - self.x.mean(axis=0)) / self.x.std(axis=0)


class Backpropagation:
    def __init__(self, train_feature):
        self.num_of_layer = None
        self.y_hat = []
        self.w = []
        self.b = []
        self.v = []
        self.fx_dash = []
        self.num_of_nodes_per_layer = []
        self.mu = 0.01
        self.train_feature = train_feature
        self.predicted_class_value = 0

    def initialize_network(self, line):
        self.num_of_layer = int(len(line)) + 1
        self.num_of_nodes_per_layer.append(self.train_feature.num_of_features)
        for i in range(len(line)):
            self.num_of_nodes_per_layer.append(int(line[i]))
        self.num_of_nodes_per_layer.append(self.train_feature.number_of_classes)

    def set_random_weight(self):
        np.random.seed(1)
        for i in range(self.num_of_layer):
            w = np.random.randn(self.num_of_nodes_per_layer[i + 1], self.num_of_nodes_per_layer[i])
            self.w.append(w)
        self.b = np.zeros((1, self.num_of_layer))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_diff(self, fx):
        return fx * (1 - fx)

    def calculate_del_for_layer_L(self, sample_no, layer_no):
        del_L = np.subtract(self.y_hat[sample_no][layer_no], self.train_feature.y[sample_no])
        del_L = np.multiply(del_L, self.fx_dash[sample_no][layer_no])
        return del_L

    def calculate_del_for_layer_less_than_L(self, prev_del, sample_no, layer_no):
        del_r = np.dot(self.w[layer_no], np.diagflat(self.fx_dash[sample_no][layer_no]))
        del_r = np.dot(prev_del.T, del_r)
        return del_r.T

    def predict(self, y_hat, actual_class):
        self.predicted_class_value = 0
        for i in range(self.train_feature.number_of_classes):
            if y_hat[self.predicted_class_value][0] < y_hat[i][0]:
                self.predicted_class_value = i
        if actual_class == self.predicted_class_value + 1:
            return True
        else:
            return False

    def calculate_error(self, y_hat, y):
        err = np.subtract(y_hat, y)
        err = np.square(err)
        return np.sum(err)

    def forward_propagation(self, itr):
        error = 0.0
        no_of_success = 0
        for i in range(self.train_feature.num_of_sample):
            v = []
            y = [self.train_feature.x[i]]
            fx_dash = [self.sigmoid_diff(self.train_feature.x[i])]
            for j in range(self.num_of_layer):
                v.append(np.dot(self.w[j], y[j]) + self.b[0][j])
                y.append(self.sigmoid(np.dot(self.w[j], y[j]) + self.b[0][j]))
                fx_dash.append(self.sigmoid_diff(y[j + 1]))
            if self.predict(y[self.num_of_layer], self.train_feature.class_no[i]):
                no_of_success += 1
            error += self.calculate_error(y[self.num_of_layer], self.train_feature.y[i])
            self.v.append(v)
            self.y_hat.append(y)
            self.fx_dash.append(fx_dash)
        # if itr % 100 == 0:
        #     print(itr, "---->", error)

    def back_propagation(self):
        del_r = None
        w_new = copy.deepcopy(self.w)
        for i in range(self.train_feature.num_of_sample):
            for j in range(self.num_of_layer, 0, -1):
                if j == self.num_of_layer:
                    del_r = self.calculate_del_for_layer_L(i, j)
                else:
                    del_r = self.calculate_del_for_layer_less_than_L(del_r, i, j)
                w_new[j - 1] -= self.mu * np.dot(del_r, self.y_hat[i][j - 1].T)
        self.w = w_new

    def train(self):
        for i in range(500):
            self.forward_propagation(i)
            self.back_propagation()
            self.v.clear()
            self.y_hat.clear()
            self.fx_dash.clear()

    def test(self, test_feature):
        num_of_success = 0
        error = 0
        for i in range(test_feature.num_of_sample):
            y = [test_feature.x[i]]
            for j in range(self.num_of_layer):
                y.append(self.sigmoid(np.dot(self.w[j], y[j]) + self.b[0][j]))
            if self.predict(y[self.num_of_layer], test_feature.class_no[i]):
                num_of_success += 1
            else:
                print(i + 1, test_feature.x[i].T, test_feature.class_no[i], self.predicted_class_value + 1)
            error += self.calculate_error(y[self.num_of_layer], test_feature.y[i])
        print("Test Accuracy: ", (num_of_success / test_feature.num_of_sample) * 100,  "%")
        print("Error Value: ",  error)
        print("Number of layer: ", self.num_of_layer)
        print("Network Structure: ", self.num_of_nodes_per_layer, "\n")


if __name__ == '__main__':
    networkFile = "network.txt"
    trainFileName = "trainNN.txt"
    testFileName = "testNN.txt"

    train_feature = Features()
    train_feature.count_num_of_sample(trainFileName)
    train_feature.read_feature_value(trainFileName)
    train_feature.normalize_input()

    test_feature = Features()
    test_feature.count_num_of_sample(testFileName)
    test_feature.read_feature_value(testFileName)
    test_feature.normalize_input()

    file = open(networkFile, "r")
    line = file.readline().split()
    while len(line) != 0:
        backprop = Backpropagation(train_feature)
        backprop.initialize_network(line)
        backprop.set_random_weight()
        backprop.train()
        backprop.test(test_feature)
        line = file.readline().split()
    file.close()
