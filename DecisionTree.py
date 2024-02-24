import csv
import numpy as np
from sklearn.preprocessing import StandardScaler

class DecisionNode:
    """Class to represent a single node in the decision tree."""

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

        # for leaf node
        self.value = value


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        # stopping criteria
        if depth >= self.max_depth or num_labels == 1:
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)

        feature_idx, threshold = self._best_split(X, y, n_features)
        if feature_idx is None:
            return DecisionNode(value=self._most_common_label(y))

        left_idxs, right_idxs = self._split(X[:, feature_idx], threshold)
        left = self._grow_tree(X[left_idxs,:], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs,:], y[right_idxs], depth+1)
        return DecisionNode(feature_index=feature_idx, threshold=threshold, left=left, right=right)

    def _best_split(self, X, y, n_features):
        best_feature, best_threshold = None, None
        best_gini = 1.0  # maximum possible value
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gini = self._gini_index(X[:, feature_index], y, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold

    def _gini_index(self, X_feature, y, threshold):
        # split dataset
        left_idxs, right_idxs = self._split(X_feature, threshold)
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        if n_left == 0 or n_right == 0:
            return 1.0

        # compute gini for each child
        unique_labels_left = np.unique(y[left_idxs])
        gini_left = 1.0 - sum((np.sum(y[left_idxs] == c) / n_left)**2 for c in unique_labels_left)

        unique_labels_right = np.unique(y[right_idxs])
        gini_right = 1.0 - sum((np.sum(y[right_idxs] == c) / n_right)**2 for c in unique_labels_right)

        # weighted gini
        return (n_left / n) * gini_left + (n_right / n) * gini_right



    def _split(self, X_feature, threshold):
        left_idxs = np.argwhere(X_feature <= threshold).flatten()
        right_idxs = np.argwhere(X_feature > threshold).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    


testfile = open('test.csv')
trainfile = open('train.csv')

test_csvreader = csv.reader(testfile)
train_csvreader = csv.reader(trainfile)

train=[]
train_ids = []
train_labels = []

header=next(train_csvreader)

for row in train_csvreader:
    float_row = [float(item) for item in row]
    train_ids.append(float_row[0])
    train.append(float_row[1:7])
    train_labels.append(float_row[7])

test=[] 
test_ids = []

header=next(test_csvreader)

for row in test_csvreader:
    float_row = [float(item) for item in row]
    test_ids.append(float_row[0])
    test.append(float_row[1:7])
    
train = np.array(train)
test = np.array(test)

#--------------------------
scaler = StandardScaler()
train = scaler.fit_transform(train)

clf = DecisionTree(max_depth=5)
clf.fit(train, np.array(train_labels).astype(int))

y_pred = clf.predict(test)
#--------------------------

output_formatat = []
for i in range(len(test)):
    output_formatat.append(str(test_ids[i]) + ',' + str(y_pred[i]))


np.savetxt('submission2.csv', output_formatat, "%s", header="id,is_anomaly", comments="")




