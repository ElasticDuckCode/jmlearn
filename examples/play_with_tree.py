import time
import numpy as np
import graphviz
from scipy import stats

from sklearn import tree
from sklearn.datasets import load_iris

from jmlearn.tree import BinaryNode, DecisionTree, view_tree


def main():

    iris = load_iris()
    data, labels = iris.data, iris.target
    order = np.arange(data.shape[0])
    np.random.shuffle(order)
    data = data[order]
    labels = labels[order]

    # Training
    data = data[50:]
    labels = labels[50:]

    print("Training Jake's Decision Tree...", end="", flush=True)
    dt = DecisionTree()
    st = time.process_time()
    dt.fit(data, labels, height=3)
    et = time.process_time()
    print(f"done!\n\tTook {(et-st)*1e3:.3f}ms", flush=True)

    dot = dt.root.view_tree()
    dot.render(directory="/tmp/jmlearn")

    print("Training Scikit's Decision Tree...", end="", flush=True)
    skdt = tree.DecisionTreeClassifier(max_depth=3)
    st = time.process_time()
    skdt = skdt.fit(data, labels)
    et = time.process_time()
    print(f"done!\n\tTook {(et-st)*1e3:.3f}ms", flush=True)

    dot = graphviz.Source(tree.export_graphviz(skdt, out_file=None))
    dot.render(directory="/tmp/sklearn")

    # Test
    data = data[:50]
    labels = labels[:50]

    pred = dt(data)
    acc = (pred == labels).mean()
    print(f"jmlearn Test Accuracy: {acc:.2%}")

    pred = skdt.predict(data)
    acc = (pred == labels).mean()
    print(f"sklearn Test Accuracy: {acc:.2%}")

    return


if __name__ == "__main__":
    main()
