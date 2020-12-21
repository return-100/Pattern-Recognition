import numpy as np
import math as math
from numpy.linalg import inv
from numpy.linalg import det


class BayesianClassifier:
    def __init__(self, class_no, num_of_feature, total_num_of_sample):
        self.class_no = class_no
        self.feature_val = []
        self.std_arr = []
        self.mean_arr = []
        self.num_of_feature = num_of_feature
        self.num_of_sample = 0
        self.total_num_of_sample = total_num_of_sample
        for i in range(num_of_feature):
            self.feature_val.append([])

    def read_feature_value(self):
        f = open("Train.txt", "r")
        line = f.readline()
        while line:
            line = f.readline().split()
            if len(line) == 0:
                break
            if self.class_no == int(line[self.num_of_feature]):
                self.num_of_sample += 1
                for i in range(self.num_of_feature):
                    self.feature_val[i].append(float(line[i]))
        f.close()

    def calculate_std_mean(self):
        for i in range(self.num_of_feature):
            self.std_arr.append(np.std(self.feature_val[i]))
            self.mean_arr.append(np.mean(self.feature_val[i]))

    def get_probability(self, line):
        ret = (self.num_of_sample / self.total_num_of_sample)
        for i in range(self.num_of_feature):
            denom = math.sqrt(2.0 * 3.1416 * self.std_arr[i] * self.std_arr[i])
            val1 = float(line[i]) - self.mean_arr[i]
            val2 = 2.0 * self.std_arr[i] * self.std_arr[i]
            nom = math.exp(-math.pow(val1, 2) / val2)
            ret *= (nom / denom)
        return ret


class BayesianClassifierMultivariant:
    def __init__(self, class_no, num_of_feature, total_num_of_sample):
        self.class_no = class_no
        self.feature_val = []
        self.std_arr = []
        self.mean_arr = []
        self.cov_mat = []
        self.num_of_feature = num_of_feature
        self.num_of_sample = 0
        self.total_num_of_sample = total_num_of_sample
        for i in range(num_of_feature):
            self.feature_val.append([])
            self.cov_mat.append([])

    def read_feature_value(self):
        f = open("Train.txt", "r")
        line = f.readline()
        while line:
            line = f.readline().split()
            if len(line) == 0:
                break
            if self.class_no == int(line[self.num_of_feature]):
                self.num_of_sample += 1
                for i in range(self.num_of_feature):
                    self.feature_val[i].append(float(line[i]))
        f.close()

    def fill_cov_mat(self):
        for i in range(self.num_of_feature):
            val = 0.0
            for j in range(self.num_of_feature):
                for k in range(self.num_of_sample):
                    val = val + (self.feature_val[i][k] - self.mean_arr[i]) * (self.feature_val[j][k] - self.mean_arr[j])
                val = val / float(self.num_of_sample)
                self.cov_mat[i].append(val)

    def calculate_std_mean(self):
        for i in range(self.num_of_feature):
            self.std_arr.append(np.std(self.feature_val[i]))
            self.mean_arr.append(np.mean(self.feature_val[i]))

    def get_probability(self, line):
        prior_probability = (self.num_of_sample / self.total_num_of_sample)
        temp_mat = []
        for i in range(self.num_of_feature):
            temp_mat.append([])
            temp_mat[i].append(float(line[i]) - self.mean_arr[i])
        #print(np.array(self.cov_mat).shape)
        scalar = np.dot(np.array(temp_mat).T, inv(self.cov_mat))
        scalar = np.dot(scalar, temp_mat)
        nom = math.exp(-0.5 * scalar[0][0])
        denom = math.pow(2 * 3.1416, self.num_of_feature / 2.0) * math.pow(det(self.cov_mat), .5)
        return (prior_probability * (nom / denom))


if __name__ == "__main__":
    f = open("train.txt", "r")
    line = f.readline().split()
    num_of_feature = int(line[0])
    num_of_class = int(line[1])
    num_of_sample = int(line[2])
    f.close()
    class_list = []
    for i in range(num_of_class):
        class_list.append(BayesianClassifierMultivariant(i + 1, num_of_feature, num_of_sample))
        class_list[i].read_feature_value()
        class_list[i].calculate_std_mean()
        class_list[i].fill_cov_mat()
    f = open("test.txt", "r")
    total = 0
    success = 0
    line = f.readline().split()
    while len(line) != 0:
        mx = 0.0
        current_class = 1
        total += 1
        for i in range(num_of_class):
            if mx < class_list[i].get_probability(line):
                mx = class_list[i].get_probability(line)
                current_class = i + 1
        if current_class == int(line[num_of_feature]):
            success += 1
        else:
            print(total, end='')
            for k in range(num_of_feature):
                print(" ", line[k], end='')
            print(" ", line[num_of_feature], end='')
            print(" ", current_class)
        line = f.readline().split()
    print((success / total) * 100)