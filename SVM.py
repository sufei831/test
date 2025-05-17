import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# 生成toy_data
def toy_data():
    X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1,
                               n_informative=1, n_redundant=0, n_repeated=0, random_state=42)
    return X, y


# 生成random_data
def random_data():
    X, y = make_classification(n_features=2, n_classes=2, n_redundant=0)
    return X, y


# 生成iris_data
def iris_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    return X, y


# 1: 不同核函数和不同平衡系数C
def kernel_C(X, y):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    Cs = [0.1, 1, 10]

    for kernel in kernels:
        for C in Cs:
            clf = SVC(kernel=kernel, C=C)
            clf.fit(X, y)
            print("Kernel:", kernel, "| C:", C, "| Accuracy:", clf.score(X, y))


# 2: 最佳超参数配置
def find_best_parameters(X_train, y_train, X_test, y_test):
    best_accuracy = 0
    best_params = None

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    Cs = [0.1, 1, 10]

    for kernel in kernels:
        for C in Cs:
            clf = SVC(kernel=kernel, C=C)
            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (kernel, C)

    print("Best parameters:", best_params)
    print("Best accuracy:", best_accuracy)


# 可视化
def draw(X, y, clf):
    plt.figure(figsize=(10, 8))

    # 数据
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=100, facecolors='none', edgecolors='k', linewidths=2)

    # 支持向量
    sv = clf.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], marker='o', s=200, facecolors='none', edgecolors='k', linewidths=2)
    plt.scatter(sv[:, 0], sv[:, 1], marker='o', s=100, facecolors='none', edgecolors='k', linewidths=1)

    # 决策边界
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6, levels=[-100, 0, 100])
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])

    plt.show()


if __name__ == '__main__':
    # 1: toy_data()
    print("Experiment 1: toy_data()")
    X_toy, y_toy = toy_data()
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_toy, y_toy, test_size=0.1, random_state=42)
    kernel_C(X_toy, y_toy)

    # 2: iris_data()
    print("\nExperiment 2: iris_data()")
    X_iris, y_iris = iris_data()
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_iris, y_iris, test_size=0.1, random_state=42)
    find_best_parameters(X_train2, y_train2, X_test2, y_test2)

    # 3: random_data()
    print("\nExperiment 3: random_data()")
    X_random, y_random = random_data()
    X_train, X_test, y_train, y_test = train_test_split(X_random, y_random, test_size=0.1, random_state=42)
    best_kernel = 'linear'
    best_C = 1
    best_clf = SVC(kernel=best_kernel, C=best_C)
    best_clf.fit(X_train, y_train)
    draw(X_random, y_random, best_clf)
