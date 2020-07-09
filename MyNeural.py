import numpy as np
import random
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoidDerivationx(y):
    return y * (1 - y)

if __name__ == '__main__':
    #初始化一些参数
    alpha = 0.5
    w1 =np.array([[random.random(), random.random()],[random.random(),random.random()],[random.random(),random.random()],[random.random(),random.random()],[random.random(), random.random()]])  #Weight of input layer
    # 输出层的权重
    w2 = np.array([[random.random(),random.random(),random.random(),random.random(),random.random()]]) 
    # 偏置，统一这个
    b = 1
    training_sets = [
    [[0, 0], [0]],
    [[0, 1], [0]],
    [[1, 0], [0]],
    [[1, 1], [1]]
    ]
    
           # 前向传播
        # 隐含层
    z1 = np.dot(training_sets[0][0], w1.transpose()) + b
        # 隐含层输出
    a1 = sigmoid(z1)

    z2 = np.dot(a1,w2.transpose())  + b
        # 输出层的输出
    a2 = sigmoid(z2)
    for i in range(5000):
        for train in training_sets:
            # print(train)
            x = train[0]
            y = train[1]         
            delta2 = np.multiply(-(y-a2), np.multiply(a2, 1-a2))
        # delta1 = np.multiply(np.dot(np.array(w2).T, delta2), np.multiply(a1, 1-a1))
            delta1 = np.multiply(np.dot(delta2, w2), np.multiply(a1, 1-a1))    
            for i in range(len(w2)):
                w2[i] = w2[i] - alpha * delta2[i] * a1

            for i in range(len(w1)):
                w1[i] = w1[i] - alpha * delta1[i] * np.array(x)

            #继续前向传播，算出误差值
            z1 = np.dot(w1, x) + b
            a1 = sigmoid(z1)
            z2 = np.dot(w2, a1) + b
            a2 = sigmoid(z2)
            print("真实值"+str(y)+"---预测"+str(a2))
    # print(w1)

 
