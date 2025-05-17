import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# 绘制边界
def draw_boundary():
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', s=100)


# 绘制决策边界
def decision_boundary():
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建网格
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制等高线
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()


# 绘制支持向量
def draw_support_vector(clf):
    sv = clf.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], marker='s', s=200, facecolors='none')

if __name__ == '__main__':
    # 创建示例数据
    X = np.array([[-1, -3], [-3, -1], [3, 2], [1, 2]])
    y = np.array([1, 1, 2, 2])

    # 创建并训练模型
    clf = SVC(kernel='linear')
    clf.fit(X, y)

    # 可视化过程
    draw_boundary()
    draw_support_vector(clf)
    decision_boundary()
