############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Grey Wolf Optimizer

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Grey_Wolf_Optimizer, File: Python-MH-Grey Wolf Optimizer.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Grey_Wolf_Optimizer>

############################################################################

# Required Libraries
import numpy  as np
import math
import random
import os

import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_validate
import numpy.random as rd
import matplotlib.pyplot as plt

import csv
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
import random
from sklearn.linear_model import LinearRegression

from keras import optimizers  # 优化器
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, TimeDistributed, RepeatVector
from keras.layers import Dropout
from keras import layers
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import time
# import g_data as G
# import get_error

import numpy as np

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split  #数据集的分割函数

from sklearn.preprocessing import StandardScaler,MinMaxScaler      #数据预处理
from math import sqrt
from sklearn import metrics

import pandas as pd

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



class RELM_HiddenLayer:



    """

        正则化的极限学习机

        :param x: 初始化学习机时的训练集属性X

        :param num: 学习机隐层节点数

        :param C: 正则化系数的倒数

    """



    def __init__(self, x, num, C=10):

        row = x.shape[0]

        columns = x.shape[1]

        rnd = np.random.RandomState(0)

        # 权重w

        self.w = rnd.uniform(-1, 1, (columns, num))

        # 偏置b

        self.b = np.zeros([row, num], dtype=float)

        for i in range(num):

            rand_b = rnd.uniform(-0.4,0.2)

            for j in range(row):

                self.b[j, i] = rand_b

        self.H0 = np.matrix(self.sigmoid(np.dot(x, self.w) + self.b))

        self.C = C

        self.P = (self.H0.H * self.H0 + len(x) / self.C).I

        #.T:共轭矩阵,.H:共轭转置,.I:逆矩阵



    @staticmethod

    def sigmoid(x):

        """

            激活函数sigmoid

            :param x: 训练集中的X

            :return: 激活值

        """

        return 1.0 / (1 + np.exp(-x))



    @staticmethod

    def softplus(x):

        """

            激活函数 softplus

            :param x: 训练集中的X

            :return: 激活值

        """

        return np.log(1 + np.exp(x))



    @staticmethod

    def tanh(x):

        """

            激活函数tanh

            :param x: 训练集中的X

            :return: 激活值

        """

        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))



    # 回归问题 训练

    def regressor_train(self, T):

        """

            初始化了学习机后需要传入对应标签T

            :param T: 对应属性X的标签T

            :return: 隐层输出权值beta

        """



 #       all_m = np.dot(self.P, self.H0.H)

 #       self.beta = np.dot(all_m, T)

 #       return self.beta

        all_m = np.dot(self.P, self.H0.H)

        self.beta = np.dot(all_m, T)



        return self.beta



    # 回归问题 测试

    def regressor_test(self, test_x):

        """

            传入待预测的属性X并进行预测获得预测值

            :param test_x:被预测标签的属性X

            :return: 被预测标签的预测值T

        """

        b_row = test_x.shape[0]

        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])

   #     h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])

        result = np.dot(h, self.beta)

 #       result =np.argmax(result,axis=1)

        return result

url = '3国内碳价预测（2021.09.14-2022.10.07）（没有OFR FSI和SCMP）.csv'
data = pd.read_csv(url,encoding='gbk')
data=data.dropna()

df_list = list(data.columns)[:]
print(df_list)
stdsc = MinMaxScaler()

X_data = data[df_list].values.reshape(len(data), len(df_list))
X_data = stdsc.fit_transform(X_data)
Y = data[data.columns[-1]].values.reshape(len(data), 1)
Y = stdsc.fit_transform(Y)

future =1
if future == 0:
    X_data = X_data[:, :]
else:
    X_data = X_data[:-future, :]
    Y = Y[future:, :]

split = 0.2
test_ = int(len(X_data)*split)
print(test_)
train_ = 1 - test_
train_feature = X_data[:train_]
test_feature= X_data[train_:]
train_label = Y[:train_]
test_label= Y[train_:]


# Function
def target_function(x):
    x1, x2= x[0], x[1]
    if x1<1:
        x1=1
    if x2<1:
        x2=10

    model = RELM_HiddenLayer(train_feature, int(x1),int (x2))

    model.regressor_train(train_label)
    # dotdata=tree.export_text(clf,feature_names=feature_train.columns)
    # print(dotdata)

    predict1 = model.regressor_test(test_feature)
    predict = stdsc.inverse_transform(predict1)
    target1 = stdsc.inverse_transform(test_label)

    # fitness = metrics.mean_absolute_error(target1, predict)
    fitness= r2_score(target1, predict)

    return -fitness

# Function: Initialize Variables
def initial_position(pack_size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((pack_size, len(min_values)+1))
    for i in range(0, pack_size):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

# Function: Initialize Alpha
def alpha_position(dimension = 2, target_function = target_function):
    alpha = np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        alpha[0,j] = 0.0
    alpha[0,-1] = target_function(alpha[0,0:alpha.shape[1]-1])
    return alpha

# Function: Initialize Beta
def beta_position(dimension = 2, target_function = target_function):
    beta = np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        beta[0,j] = 0.0
    beta[0,-1] = target_function(beta[0,0:beta.shape[1]-1])
    return beta

# Function: Initialize Delta
def delta_position(dimension = 2, target_function = target_function):
    delta =  np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        delta[0,j] = 0.0
    delta[0,-1] = target_function(delta[0,0:delta.shape[1]-1])
    return delta

# Function: Updtade Pack by Fitness
def update_pack(position, alpha, beta, delta):
    updated_position = np.copy(position)
    for i in range(0, position.shape[0]):
        if (updated_position[i,-1] < alpha[0,-1]):
            alpha[0,:] = np.copy(updated_position[i,:])
        if (updated_position[i,-1] > alpha[0,-1] and updated_position[i,-1] < beta[0,-1]):
            beta[0,:] = np.copy(updated_position[i,:])
        if (updated_position[i,-1] > alpha[0,-1] and updated_position[i,-1] > beta[0,-1]  and updated_position[i,-1] < delta[0,-1]):
            delta[0,:] = np.copy(updated_position[i,:])
    return alpha, beta, delta

# Function: Updtade Position
def update_position(position, alpha, beta, delta, a_linear_component = 2, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    updated_position = np.copy(position)
    for i in range(0, updated_position.shape[0]):
        for j in range (0, len(min_values)):   
            r1_alpha              = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            r2_alpha              = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            a_alpha               = 2*a_linear_component*r1_alpha - a_linear_component
            c_alpha               = 2*r2_alpha      
            distance_alpha        = abs(c_alpha*alpha[0,j] - position[i,j]) 
            x1                    = alpha[0,j] - a_alpha*distance_alpha   
            r1_beta               = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            r2_beta               = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            a_beta                = 2*a_linear_component*r1_beta - a_linear_component
            c_beta                = 2*r2_beta            
            distance_beta         = abs(c_beta*beta[0,j] - position[i,j]) 
            x2                    = beta[0,j] - a_beta*distance_beta                          
            r1_delta              = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            r2_delta              = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            a_delta               = 2*a_linear_component*r1_delta - a_linear_component
            c_delta               = 2*r2_delta            
            distance_delta        = abs(c_delta*delta[0,j] - position[i,j]) 
            x3                    = delta[0,j] - a_delta*distance_delta                                 
            updated_position[i,j] = np.clip(((x1 + x2 + x3)/3),min_values[j],max_values[j])     
        updated_position[i,-1] = target_function(updated_position[i,0:updated_position.shape[1]-1])
    return updated_position

# GWO Function
fitlist=[]
def grey_wolf_optimizer(pack_size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function):    
    count    = 0
    alpha    = alpha_position(dimension = len(min_values), target_function = target_function)
    beta     = beta_position(dimension  = len(min_values), target_function = target_function)
    delta    = delta_position(dimension = len(min_values), target_function = target_function)
    position = initial_position(pack_size = pack_size, min_values = min_values, max_values = max_values, target_function = target_function)
    while (count <= iterations):      
        print("Iteration = ", count, " f(x) = ", alpha[-1])
        fitlist.append(alpha[-1][2])
        a_linear_component = 2 - count*(2/iterations)
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        position           = update_position(position, alpha, beta, delta, a_linear_component = a_linear_component, min_values = min_values, max_values = max_values, target_function = target_function)       
        count              = count + 1       
    print(alpha[-1])    
    return alpha

######################## Part 1 - Usage ####################################

# Function to be Minimized (Six Hump Camel Back). Solution ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def six_hump_camel_back(variables_values = [0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value


# Function to be Minimized (Rosenbrocks Valley). Solution ->  f(x) = 0; xi = 1
def rosenbrocks_valley(variables_values = [0,0]):
    func_value = 0
    last_x = variables_values[0]
    for i in range(1, len(variables_values)):
        func_value = func_value + (100 * math.pow((variables_values[i] - math.pow(last_x, 2)), 2)) + math.pow(1 - last_x, 2)
    return func_value

gwo = grey_wolf_optimizer(pack_size = 80, min_values = [1,1], max_values = [100,200], iterations = 100, target_function = target_function)
df=pd.DataFrame(fitlist)
df.to_csv("fitness.csv")