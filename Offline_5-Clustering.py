import matplotlib.pyplot as plt
import numpy as np
import sys
import random
from copy import deepcopy
import math as math

sys.setrecursionlimit(10000)

class clusturingAlgorithm:
    def __init__(self, k_th_nearest, is_plot):
        self.n = 0
        self.eps = 0
        self.dataset = []
        self.visited = []
        self.cluster = []
        self.unvisited = set()
        self.core_points = []
        self.k = k_th_nearest
        self.minpts = k_th_nearest
        self.total_num_of_cluster = 0
        self.is_plot = is_plot

    def read_input(self, filename):
        file = open(filename, "r")
        line = file.readline().split()
        mx, my = -np.inf, -np.inf
        while True:
            if len(line) == 0:
                break
            self.dataset.append([float(line[0]), float(line[1])])
            mx = max(mx, float(line[0]))
            my = max(my, float(line[1]))
            line = file.readline().split()
        self.n = len(self.dataset)
        for i in range(self.n):
            self.dataset[i][0] /= mx
            self.dataset[i][1] /= my
            self.visited.append(False)
            self.cluster.append(0)
            self.core_points.append(False)
            self.unvisited.add(i)

    def estimate_eps(self):
        distance = []
        for i in range(self.n):
            temp = []
            for j in range(self.n):
                if i != j:
                    dist = (self.dataset[i][0] - self.dataset[j][0]) ** 2 + (
                                self.dataset[i][1] - self.dataset[j][1]) ** 2
                    temp.append(math.sqrt(dist))
            temp.sort()
            distance.append(temp[self.k - 1])
        distance.sort()
        if self.is_plot:
            x = [i for i in range(self.n)]
            plt.figure(1)
            plt.plot(x, distance)
            plt.grid()
            plt.show()
        self.eps = float(input("EPS = "))

    def dfs(self, u):
        self.visited[u] = True
        self.cluster[u] = self.total_num_of_cluster
        self.unvisited.remove(u)
        unvisited = deepcopy(self.unvisited)
        for i in unvisited:
            dist = (self.dataset[i][0] - self.dataset[u][0]) ** 2 + (self.dataset[i][1] - self.dataset[u][1]) ** 2
            if self.visited[i] is False and math.sqrt(dist) <= self.eps:
                self.dfs(i)

    def dbscan(self):
        for i in range(self.n):
            neighbors = 0
            for j in range(self.n):
                dist = (self.dataset[i][0] - self.dataset[j][0]) ** 2 + (self.dataset[i][1] - self.dataset[j][1]) ** 2
                if math.sqrt(dist) <= self.eps:
                    neighbors += 1
            if neighbors >= self.minpts:
                self.core_points[i] = True
        for i in range(self.n):
            if self.visited[i] is False and self.core_points[i] is True:
                self.total_num_of_cluster += 1
                self.dfs(i)
        print(self.total_num_of_cluster)
        colors = ['#585d8a', '#858482', '#23ccc9', '#e31712', '#91f881', '#89b84f', '#fedb00', '#0527f9', '#571d08', '#ffae00', '#b31d5b', '#702d75']
        if self.is_plot:
            for i in range(self.n):
                if self.cluster[i]:
                    plt.scatter(self.dataset[i][0], self.dataset[i][1], color=colors[self.cluster[i] - 1])
            plt.show()

    def plus_plus(self, random_state=42):
        np.random.seed(random_state)
        centroids = [self.dataset[0]]
        for i in range(self.total_num_of_cluster):
            dist_sq = np.array([min([np.inner(c - x, c - x) for c in np.array(centroids)]) for x in np.array(self.dataset)])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break
            centroids.append(self.dataset[i])
        return centroids

    def kmeans(self):
        self.dataset.sort()
        centroid = self.plus_plus()
        print(centroid)
        for i in range(1000):
            mean_x = []
            mean_y = []
            cluster_cnt = []
            for j in range(self.total_num_of_cluster):
                mean_x.append(0)
                mean_y.append(0)
                cluster_cnt.append(0)
            for j in range(self.n):
                dist = np.inf
                for k in range(self.total_num_of_cluster):
                    temp_dist = (self.dataset[j][0] - centroid[k][0]) ** 2 + (self.dataset[j][1] - centroid[k][1]) ** 2
                    if math.sqrt(temp_dist) < dist:
                        dist = math.sqrt(temp_dist)
                        self.cluster[j] = k + 1
                cluster_cnt[self.cluster[j] - 1] += 1
                mean_x[self.cluster[j] - 1] += self.dataset[j][0]
                mean_y[self.cluster[j] - 1] += self.dataset[j][1]
            is_break = True
            for j in range(self.total_num_of_cluster):
                mean_x[j] /= cluster_cnt[j]
                mean_y[j] /= cluster_cnt[j]
                if abs(centroid[j][0] - mean_x[j]) > 0.0001 or abs(centroid[j][1] - mean_y[j]) > 0.0001:
                    is_break = False
                centroid[j][0] = mean_x[j]
                centroid[j][1] = mean_y[j]
            if is_break:
                break
        colors = ['#585d8a', '#858482', '#23ccc9', '#e31712', '#91f881', '#89b84f', '#fedb00', '#0527f9', '#571d08', '#ffae00', '#b31d5b', '#702d75']
        for i in range(self.n):
            if self.cluster[i]:
                plt.scatter(self.dataset[i][0], self.dataset[i][1], color=colors[self.cluster[i] - 1])
        plt.show()

if __name__ == "__main__":
    solve = clusturingAlgorithm(4, True)
    solve.read_input("blobs.txt")
    solve.estimate_eps()
    solve.dbscan()
    solve.kmeans()

# bisecting - eps: 0.03
# blob - eps: 0.08
# moon - eps: 0.061
