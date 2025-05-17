import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

def get_data():
    return load_diabetes(return_X_y=True)
def init_params():
    w=np.zeros(10)
    b=0.0
    return w,b

def predict(w,b,x):
    return np.dot(w,x)+b

def train(X,Y,lr=0.01):
    w,b=init_params()
    n=len(X)

    R_thetas=[]
    R_thetas.append((np.mean([np.square(predict(w,b,X[i])-Y[i])for i in range(n)])))

    for _ in range(100):
        w=w-lr*2*np.array([np.mean([(predict(w,b,X[i])-Y[i])*X[i,j]
                                    for i in range(n)])
                           for j in range(10)])
        b=b-lr*2*(np.mean([(predict(w,b,X[j])-Y[j])for j in range(n)]))
        R_thetas.append((np.mean([np.square(predict(w,b,X[i])-Y[i])for i in range(n)])))
        return w,b,R_thetas

if __name__=="__main__":
    X,Y=get_data()
    w,b,R_thetas=train(X,Y)
    print("w:",w,"b",b)
    print("R(theta):",R_thetas)

    plt.figure()
    plt.plot(R_thetas,label="R(theta)s")
    plt.xlabel("epoch")
    plt.ylabel("R(theta)")
    plt.legend()
    plt.show()