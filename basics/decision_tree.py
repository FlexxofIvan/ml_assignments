import numpy as np
import sklearn
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris


def splitter(data: np.array,
             num: int,
             threshold: float,
             feature_input: bool = False,
             tar_only: bool = False
             ) -> tuple[np.array, np.array]:
    if feature_input:
        features = data

    else:
        features = data[:, :-1]
    ind_left = features[:, num] < threshold
    left = data[ind_left]

    ind_right = features[:, num] >= threshold
    right = data[ind_right]

    if tar_only:
        return left[:, -1], right[:, -1]

    return left, right


def step_value(feature: np.array
               ) -> np.array:
    unique = np.sort(np.unique(feature))
    thresholds = (unique[:-1] + unique[1:]) / 2
    return thresholds


def entropy(class_labels: np.array
            ) -> np.array:
    unique, counts = np.unique(class_labels, return_counts=True)
    probs = counts / np.sum(counts)
    H = -sum(probs * np.log(probs))
    return H


def entropy_criteria(whole: np.array,
                     left: np.array,
                     right: np.array) -> np.array:
    H_m = entropy(whole)
    H_r = entropy(right)
    H_l = entropy(left)

    return H_m - ((len(left) / (len(whole))) * H_l) - ((len(right) / (len(whole))) * H_r)


def prob_predictor(labels: np.array) -> int:
    unique, counts = np.unique(labels, return_counts=True)
    return unique[np.argmax(counts)]


class Leaf:
    def __init__(self,
                 criterion,
                 feature_num: np.array = None,
                 threshold: np.array = None
                 ) -> None:
        self.func = criterion
        self.feature_num = feature_num
        self.threshold = threshold
        self.right = None
        self.left = None
        self.predictor = None
        self.num_cl = None

    def leaf_split(self,
                   data: np.array,
                   feature_input: bool = False
                   ) -> tuple[np.array, np.array]:
        return splitter(data=data, num=self.feature_num, threshold=self.threshold,
                        feature_input=feature_input)


class Decision_tree:

    def __init__(self,
                 criterion,
                 predictor,
                 max_depth: int,
                 min_samples: int = 2,
                 ) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.func = criterion
        self.init_leaf = None
        self.pred = predictor

    def leaf_birth(self,
                   data: np.array
                   ) -> Leaf:

        features = data[:, :-1]
        targets = data[:, -1]
        loss = 0
        feature_num = 0
        curr_thresh = 0

        for j in range(len(features[0, :])):
            threshes = step_value(feature=features[:, j])
            for thresh in threshes:
                left, right = splitter(data=data, num=j, threshold=thresh, tar_only=True)
                curr_loss = self.func(whole=targets, left=left, right=right)
                if curr_loss > loss:
                    loss = curr_loss
                    curr_thresh = thresh
                    feature_num = j

        return Leaf(criterion=self.func, feature_num=feature_num, threshold=curr_thresh)


    def recursive_fit(self, data: np.array, leaf: Leaf, current_depth) -> None:

        left, right = leaf.leaf_split(data)

        depth_cond = current_depth >= self.max_depth
        length_cond = len(left) == 0 or len(right) == 0
        min_samples_cond = len(left) <= self.min_samples or len(right) <= self.min_samples

        if depth_cond or length_cond or min_samples_cond:
            leaf.predictor = self.pred
            leaf.num_cl = self.pred(data[:, -1])
            return

        else:
            leaf.left = self.leaf_birth(left)
            self.recursive_fit(data=left, leaf=leaf.left, current_depth=current_depth + 1)

            leaf.right = self.leaf_birth(data=right)
            self.recursive_fit(data=right, leaf=leaf.right, current_depth=current_depth + 1)


    def fit(self, data: np.array) -> None:
        if self.init_leaf is None:
            self.init_leaf = self.leaf_birth(data)

        self.recursive_fit(data=data, leaf=self.init_leaf, current_depth=0)


    def rec_forward(self, data: np.array,
                    leaf: Leaf
                    ) -> np.array:

        if leaf.predictor is not None:
            predictions = np.full((data.shape[0], 1), leaf.num_cl)
            result = np.column_stack((data, predictions))
            return result

        left, right = leaf.leaf_split(data, feature_input=True)
        left_pred = self.rec_forward(data=left, leaf=leaf.left)
        right_pred = self.rec_forward(data=right, leaf=leaf.right)

        return np.concatenate([left_pred, right_pred])


    def forward(self, data: np.array) -> np.array:
        return self.rec_forward(data=data, leaf=self.init_leaf)


iris = load_iris()
features = iris.data
y = iris.target
data_combined = np.column_stack((iris.data, iris.target))
tree = Decision_tree(criterion=entropy_criteria,
                     max_depth=3,
                     min_samples=4,
                     predictor=prob_predictor)
tree.fit(data_combined)
x = tree.forward(features)

plt.scatter(x[:, 0], x[:, 1], c=x[:, -1])
plt.show()
