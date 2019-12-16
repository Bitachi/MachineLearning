#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
import matplotlib.pyplot as plt


def cos_f(x):
    '''理論値の計算'''
    return np.cos(2*np.pi*x)

def return_weight(M):   
    '''指定された次元の重みベクトルを出力'''
    #データセットを並べた行列Φの準備
    phi = np.zeros([N+1, M+1]) #N行M列の配列を用意
    for i in range(0, N+1):
        for j in range(0, M+1):
            phi[i][j] = math.pow(x[i], j) #観測値x_iのj乗をphi[i][j]に格納
    #行列Φを用いて、重みベクトルwを計算
    w = np.matmul(np.dot(np.linalg.inv((np.dot(np.transpose(phi), phi))), np.transpose(phi)), y)
    #matmul 行列積の計算
    #linalg.inv 逆行列の計算
    #transpose 転置行列の計算
    return w

def f(x, w):
    '''多項式f(x)の値を出力'''
    #xはベクトルではなくただの値
    #wはベクトル
    result = 0.0
    for i in range(0, len(w)):
        result += pow(x,i) * w[i]
    return result


def calc_E(x, w):
    '''二乗誤差を出力'''
    result = 0.0
    for i in range(0, N+1):
        result += ((f(x[i], w) - y[i])**2)/2
        result = math.sqrt((2*result)/N)
    return result



if __name__ == '__main__':
    N = 100 #データの個数
    M = [1, 2, 3, 4, 5,6, 7, 8,  9, 10] #次元
    #データセットを用意
    x = np.linspace(0, 1, N+1)
    y = []
    for i in range(N+1):
        y.append(np.cos(2*np.pi*x[i]) + np.random.normal(-0.3,0.3))

    i=1
    plt.figure(figsize=(15,12))
    for m in M:   # 多項式の次数
        w = return_weight(m)
        e = calc_E(x, w)
        plt.subplot(4,3,i)
        plt.scatter(x, y, color="orange")
        plt.plot(x,cos_f(x), label=("theotical value"), color="blue")
        plt.plot(x,f(x, w), label="predicted value", color="red")
        plt.title("dimension:" + str(m) + "\nE_rms = %.4f" % e)
        plt.subplots_adjust(hspace=0.6)
        plt.legend(loc=4)
        i+=1
    plt.show()
