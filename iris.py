import numpy as np
from math import sqrt
from random import choice
from sklearn.datasets import load_iris

class KdNode:
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt
        self.split = split
        self.left = left
        self.right = right

class KdTree:
    def __init__(self, distance):
        self.root = None
        self.distance = distance

    def create(self, data):
        k = len(data[0][0])  
        def _create_node(split, data_set):
            if not data_set: #递归出口
                return None
            data_set.sort(key=lambda x: x[0][split])
            split_pos = len(data_set) // 2
            median = data_set[split_pos]   #中位数节点
            split_next = (split + 1) % k #下一个节点
            return KdNode(
                median,
                split,
                _create_node(split_next, data_set[:split_pos]),
                _create_node(split_next, data_set[split_pos + 1:])
            )
        self.root = _create_node(0, data)

    def print(self):
        def _preorder(node: KdNode):
            print(node.dom_elt)
            if node.left:
                _preorder(node.left)
            else:
                print("#")
            if node.right:
                _preorder(node.right)
            else:
                print("#")
        _preorder(self.root)

    def find_nn(self, point):
        def _search(start_node: KdNode):
            #print("#go search")
            while start_node:
                stack.append(start_node)
                if point[start_node.split] <= start_node.dom_elt[0][start_node.split]:
                    start_node = start_node.left
                else:
                    start_node = start_node.right
            #print("#end search")

        min_node, min_dist = None, float("inf")
        if self.root is None:
            return min_node, min_dist
        stack = []
        _search(self.root)
        nodes_visited = 0
        while stack:
            node = stack.pop()
            dist = self.distance(node.dom_elt[0], point)
            nodes_visited += 1
            if dist < min_dist:
                min_dist = dist
                min_node = node
            if node.left is None and node.right is None:
                continue
            else:
                offset = abs(node.dom_elt[0][node.split] - point[node.split])
                if offset < min_dist:
                    if (point[node.split] > node.dom_elt[0][node.split]) and node.left:
                        _search(node.left)
                    elif (point[node.split] <= node.dom_elt[0][node.split]) and node.right:
                        _search(node.right)
        return min_node, min_dist

def test(kd_tree, point):
    nearest, _ = kd_tree.find_nn(point)
    return nearest.dom_elt[1]

def evaluate(kd_tree, X_valid, y_valid):
    correct = 0
    for i in range(len(X_valid)):
        if test(kd_tree, X_valid[i]) == y_valid[i]:
            correct += 1
    return correct / len(X_valid)

def train_valid_split(X, y, train_size=0.8):
    assert len(X) == len(y)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    train_end = int(train_size * len(X))
    train_indices = indices[:train_end]
    valid_indices = indices[train_end:]

    X_train, X_valid = X[train_indices], X[valid_indices]
    y_train, y_valid = y[train_indices], y[valid_indices]
    return X_train, X_valid, y_train, y_valid




if __name__=="__main__":
    # data=[[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    # kd_tree=KdTree(lambda x,y:sqrt(sum((p1-p2)**2 for p1,p2 in zip(x,y))))
    # kd_tree.create(data)
    # kd_tree.print()
    #
    # point=[2.1,3.1]
    # test(kd_tree,point)
    #
    # print("brute-forde checking:")
    # dists=[sqrt(sum((p1-p2)**2 for p1,p2 in zip(point,d)))for d in data]
    # min_dist=min(dists)
    # min_idx=dists.index(min_dist)
    # print(data[min_idx],min_dist)

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_valid, y_train, y_valid = train_valid_split(X, y, train_size=0.8)
    data = [(X_train[i], y_train[i]) for i in range(len(X_train))]
    kd_tree = KdTree(lambda x, y: sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(x, y))))
    kd_tree.create(data)
    kd_tree.print()

    accuracy = evaluate(kd_tree, X_valid, y_valid)
    print("Accuracy:", accuracy)