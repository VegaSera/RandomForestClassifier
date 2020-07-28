# Requirements for this project:

"""
Decision trees:
    Each decision tree takes its data and splits it into branches that have the greatest ability to predict
    Each decision tree will have certain parameters
        x input data
        y input target data
        number of features
        min_split - Minimum number of row samples to be able to cause a split
        max_depth - Max number of splits within each tree

    Decision trees train by splitting the data into two halves (recursion might be good here)
    Splits are based off of Gini Impurity


Random Forest:
    The methodology behind random forests is that the average error of a large number of random errors is zero.
    Random forest will need certain parameters.
        x input data
        y input target data
        number of features
        number of trees
        sample size
        depth
        minimum amount of leaves
        random seed
    Each decision tree in the forest should receive a random subset of features (feature bagging) and a random set of
    rows.
"""
import numpy as np
import pandas as pd


class DecisionTreeClassifier:
    def __init__(self, num_features, feature_indexes, row_indexes, depth=10, min_split=5):
        self.n_features = num_features
        self.feature_indexes = feature_indexes
        self.row_indexes = row_indexes
        self.depth = depth
        self.min_split = min_split
        self.score = float('inf') #Since we're looking for the lowest score on each split in each tree, we initialize to the highest possible value, infinity.

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.val = y.unique()
        self.number_rows = len(self.row_indexes)

        for i in self.feature_indexes:
            v, w = self.x.values[self.row_indexes, i], self.y[self.row_indexes]

            #Sorting the values by the feature.
            sort_index = np.argsort(v)
            sort_x, sort_y = v[sort_index], w[sort_index]

            #Initializing values before the comparator.
            right_count = self.number_rows
            right_sum = sort_y.sum()
            right_sqsum = (sort_y**2).sum()
            left_count = 0
            left_sum = 0
            left_sqsum = 0

            for j in range(0, self.number_rows- self.min_split-1):
                x_j, y_j = sort_x[j], sort_y[j]
                left_count += 1
                right_count -= 1
                left_sum += y_j
                right_sum -= y_j
                left_sqsum += y_j**2
                right_sqsum -= y_j**2

                if j < self.min_split or x_j == sort_x[j+1]:
                    continue

                left_std = std_aggregate(left_count, left_sum, left_sqsum)
                right_std = std_aggregate(right_count, right_sum, right_sqsum)
                current_score = (left_std*left_count) + (right_std*right_count)
                if current_score < self.score:
                    self.var_index = i
                    self.score = current_score
                    self.split = x_j

        #Base case, tree is a leaf
        if self.score == float('inf') or self.depth <= 0:
            return

        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        lf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        rf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        self.left_path = DecisionTreeClassifier(self.n_features, lf_idxs, self.row_indexes[lhs], depth=self.depth - 1,
                                min_split=self.min_split)
        self.right_path = DecisionTreeClassifier(self.n_features, rf_idxs, self.row_indexes[rhs], depth=self.depth - 1,
                                min_split=self.min_split)
        self.left_path.fit(self.x, self.y)
        self.right_path.fit(self.x, self.y)

    def predict(self, input):
        return np.array([self.predict_row(x) for x in input])

    def predict_row(self, input):
        if self.score == float('inf') or self.depth <= 0: # Is a leaf
            return self.val
        if input[self.var_index] <= self.split:
            path = self.left_path
        else:
            path = self.right_path
        return path.predict_row(input)

    def split_col(self):
        return self.x.values[self.row_indexes, self.var_index]



class RandomForestClassifier:
    def __init__(self, n_features=None, sample_size=None, n_trees=100, depth=10, min_leaf=5, random_seed=1337):
        np.random.seed(random_seed)

        #TODO Use SKLearn's method of specifying auto, sqrt, log2, int or float
        self.n_features = n_features

        #TODO Use SKLearn's method of specifying None, Int or Float
        self.sample_size = sample_size

        self.depth = depth
        self.min_leaf = min_leaf
        self.n_trees = n_trees

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.trees = [self.create_tree() for _ in range(self.n_trees)]
        for tree in self.trees:
            tree.fit(x, y)

    def create_tree(self):
        row_indexes = np.random.permutation(len(self.y))[:self.sample_size] #Randomly permutes the rows up to the sample size
        feature_indexes = np.random.permutation(self.x.shape[1])[:self.n_features] #Randomly permutes the features up to the feature limit.
        return DecisionTreeClassifier(self.n_features, feature_indexes, row_indexes)

    def predict(self):
        pass

def std_aggregate(count, x1, x2):
    return ((x2/count) - (x1/count)**2)**0.5