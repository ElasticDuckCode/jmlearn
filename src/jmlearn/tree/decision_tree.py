import numpy as np
from scipy import stats
from .node import BinaryNode


def class_probabilities(data, labels):
    classes = np.unique(labels)
    p = np.zeros(len(classes))
    n = data.shape[0]
    for i, label in enumerate(classes):
        k = data[label == labels, :].shape[0]
        p[i] = k / n
    return p


def get_partitions(data): 
    """Split data examples by their midpoints for each feature.
    """
    sort = np.sort(data, axis=0)
    return (sort[:-1] + sort[1:]) / 2


def split(data, labels, attribute, threshold):
    group_a = data[:, attribute] <= threshold
    group_b = np.invert(group_a)
    data_a, labels_a = data[group_a], labels[group_a]
    data_b, labels_b = data[group_b], labels[group_b]
    return (data_a, labels_a), (data_b, labels_b)


def entropy(data, labels):
    p = class_probabilities(data, labels)
    return stats.entropy(p)


def gini_impurity(data, labels):
    p = class_probabilities(data, labels)
    return 1 - p @ p


# TODO: move splits to single function, and just change out the measure that is used.
def score_partiton(attribute, thresh, data, labels, measure):
    n_points = data.shape[0]
    i_curr = measure(data, labels)
    group_a, group_b = split(data, labels, attribute, thresh)
    data_a, labels_a = group_a
    data_b, labels_b = group_b

    i_a = measure(data_a, labels_a)
    i_b = measure(data_b, labels_b)

    # calculate relative frequency of left or right
    p_l = data_a.shape[0] / n_points
    p_r = data_b.shape[0] / n_points
    
    return i_curr - p_l * i_a - p_r * i_b


def information_gain_split(data, labels):
    """Split dataset according to largest information gain
    https://en.wikipedia.org/wiki/Decision_tree_learning
    """
    partions = get_partitions(data)
    scores = np.zeros_like(partions)

    for idx, thresh in np.ndenumerate(partions):
        scores[idx] = score_partiton(idx[1], thresh, data, labels, entropy)
        
    idx = np.unravel_index(scores.argmax(), scores.shape)
    thresh = partions[idx]
    return *split(data, labels, idx[1], thresh), idx[1], thresh
    pass


def gini_split(data, labels):
    """Split dataset according to largest change in Gini Impurity
    https://en.wikipedia.org/wiki/Decision_tree_learning
    """
    partions = get_partitions(data)
    scores = np.zeros_like(partions)
        
    for idx, thresh in np.ndenumerate(partions):
        scores[idx] = score_partiton(idx[1], thresh, data, labels, gini_impurity)

    idx = np.unravel_index(scores.argmax(), scores.shape)
    thresh = partions[idx]
    return *split(data, labels, idx[1], thresh), idx[1], thresh


def CART(data, labels, height=20):

    # check our stopping criteria
    classes = np.unique(labels)

    if len(classes) == 1: # if all classes same, we can make decision
        node = BinaryNode(classes[0])

    elif height <= 0: # max tree height, decide class who is most frequent
        p = class_probabilities(data, labels)
        node = BinaryNode(classes[p.argmax()])
        
    # Split dataset according to Gini Splitting Rule (CART Algorithm)
    else:
        group_a, group_b, attribute, thresh = gini_split(data, labels)
        node = BinaryNode((attribute, thresh))
        node.left = CART(*group_a, height=height-1)
        node.right = CART(*group_b, height=height-1)

    return node


def C4_5(data, labels, height=20):

    # check our stopping criteria
    classes = np.unique(labels)

    if len(classes) == 1: # if all classes same, we can make decision
        node = BinaryNode(classes[0])

    elif height <= 0: # max tree height, decide class who is most frequent
        p = class_probabilities(data, labels)
        node = BinaryNode(classes[p.argmax()])
        
    # Split dataset according to Information Gain (C4.5 Algorithm)
    else:
        group_a, group_b, attribute, thresh = information_gain_split(data, labels)
        node = BinaryNode((attribute, thresh))
        node.left = C4_5(*group_a, height=height-1)
        node.right = C4_5(*group_b, height=height-1)

    return node


class DecisionTree:
    classification_algos = {
        'CART': CART,
        'C4.5': C4_5
    }
    
    def __init__(self, mode='classification', algo='CART'):
        self._root = None
        self._mode = mode
        self._algo = algo
        pass
    
    @property
    def root(self):
        return self._root
    
    def fit(self, data, labels, height=20):
        if self._mode == 'classification':
            self._root = self.classification_algos[self._algo](data, labels, height)

    def __call__(self, data):
        
        #TODO: Inefficient looping
        predicted_class = []
        for i, point  in enumerate(data):
            current_node = self._root
            while current_node.left is not None and current_node.right is not None:
                attribute, thresh = current_node.value
                if point[attribute] <= thresh:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            predicted_class.append(current_node.value)
        return np.asarray(predicted_class)