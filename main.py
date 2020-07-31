import numpy as np
import pandas as pd
import time

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
        self.y = np.array(y)
        self.val = pd.Series(self.y[self.row_indexes]).value_counts().index[0]
        self.number_rows = len(self.row_indexes)

        for i in self.feature_indexes:
            v = self.x.values[self.row_indexes, i]
            w = self.y[self.row_indexes]

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

        g = self.split_col()
        lhs = np.nonzero(g <= self.split)[0]
        rhs = np.nonzero(g > self.split)[0]
        lf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        rf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        self.left_path = DecisionTreeClassifier(self.n_features, lf_idxs, self.row_indexes[lhs], depth=self.depth - 1,
                                min_split=self.min_split)
        self.right_path = DecisionTreeClassifier(self.n_features, rf_idxs, self.row_indexes[rhs], depth=self.depth - 1,
                                min_split=self.min_split)
        self.left_path.fit(self.x, self.y)
        self.right_path.fit(self.x, self.y)


    def predict(self, input):
        return np.array([self.predict_row(x) for x in input.iterrows()])

    def predict_row(self, input):
        if self.score == float('inf') or self.depth <= 0: # Is a leaf
            return self.val
        if input[1][self.var_index] <= self.split:
            path = self.left_path
        else:
            path = self.right_path
        return path.predict_row(input)

    def split_col(self):
        return self.x.values[self.row_indexes, self.var_index]



class RandomForestClassifier:
    def __init__(self, n_features=None, sample_size=None, n_trees=100, depth=10, min_leaf=5, random_seed=1333):
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
        self.n_features = x.shape[1]
        self.trees = [self.create_tree() for _ in range(self.n_trees)]
        for tree in self.trees:
            tree.fit(x, y)

    def create_tree(self):
        row_indexes = np.random.permutation(len(self.y))[:self.sample_size] #Randomly permutes the rows up to the sample size
        feature_indexes = np.random.permutation(self.x.shape[1])[:self.n_features] #Randomly permutes the features up to the feature limit.
        return DecisionTreeClassifier(self.n_features, feature_indexes, row_indexes)

    def predict(self, test):
        holding_list = []
        for tree in self.trees:
            holding_list.append(tree.predict(test))
        #Aggregate results
        fin_array = []
        for col in range(len(holding_list[0])):
            dict = {}
            for row in holding_list:
                if row[col] not in dict.keys():
                    dict[row[col]] = 0
                dict[row[col]] += 1
            fin_array.append(max(dict.items(), key= lambda x: x[1])[0])

        return pd.Series(fin_array)

def std_aggregate(count, x1, x2):
    return ((x2/count) - (x1/count)**2)**0.5

def accuracy_score(y_pred, y_true):
    y_pred = list(y_pred)
    y_true = list(y_true)
    if len(y_pred) == len(y_true):
        correct = 0
        total = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                correct += 1
            total += 1
        return correct/total

    else:
        raise ValueError(f"Length of series must be equal. {y_pred} != {y_true}")


if __name__ == '__main__':
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df[4] = df[4].map({"Iris-setosa":0, "Iris-virginica":1, "Iris-versicolor":2})
    df.columns = ['sepal1', 'sepal2', 'petal1', 'petal2', 'classification']

    my_model_acc = []
    sk_model_acc = []
    for i in range(200):
        starttime = time.time()
        my_model = time.time()
        n_trees = 100
        max_depth = 20
        min_leaf = 5
        random_state = i

        model = RandomForestClassifier(n_trees=n_trees, depth=max_depth, min_leaf=min_leaf, random_seed=random_state)
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df)
        features = df.columns[:-1]
        target = df.columns[-1]
        x_train = train[features]
        x_test = test[features]
        y_train = train[target]
        y_test = test[target]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        my_model_end = time.time()
        sk_start = time.time()

        from sklearn.ensemble import RandomForestClassifier as RFC
        skmodel = RFC(max_depth=max_depth, min_samples_leaf=min_leaf, n_estimators=n_trees, random_state=random_state)
        skmodel.fit(x_train, y_train)

        y_pred2 = skmodel.predict(x_test)

        from sklearn.metrics import accuracy_score as acc_score
        my_model_acc.append(acc_score(y_test, y_pred))
        sk_model_acc.append(acc_score(y_test, y_pred2))
        endtime = time.time()
        print(f"Finished iteration {i} in {endtime-starttime:.2f} seconds. My model took {my_model_end-my_model:.2f} seconds. SKLearn took {endtime-sk_start:.2f} seconds.")
        #
        # print("My model's accuracy", acc_score(y_test, y_pred))
        # print("SKLearn's model's accuracy", acc_score(y_test, y_pred2))

    print("Model accuracy over 100 seeds")
    print("My model's average accuracy - ", (sum(my_model_acc)/len(my_model_acc)))
    print("SKLearn's model's average accuracy - ", (sum(sk_model_acc) / len(sk_model_acc)))






