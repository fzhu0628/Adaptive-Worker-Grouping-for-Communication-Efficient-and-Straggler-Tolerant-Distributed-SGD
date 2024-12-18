# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 11:50:15 2021

@author: Feng Zhu
"""
import sys, os
import time
from common.layers import *
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
import math
import operator
from scipy import stats
import matplotlib.pyplot as plt
from pylab import *
from common.optimizer import *
import copy
import collections
class SGD:

    """随机梯度下降法（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        params -= self.lr * grads 
class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
        #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
        self.m += (1 - self.beta1) * (grads - self.m)
        self.v += (1 - self.beta2) * (grads**2 - self.v)
        
        params -= lr_t * self.m / (np.sqrt(self.v) + 1e-7)
        
        #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
        #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
        #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

class CADA:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.v_h = None
        self.temp = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            self.v_h = np.zeros_like(params)
            self.temp = np.zeros_like(params)
        self.temp = self.v_h
        lr_t  = self.lr         
        
        self.v_h = np.array(max(self.temp.tolist(), self.v.tolist()))
        self.m = self.beta1*self.m + (1-self.beta1)*grads
        self.v = self.beta2*self.v_h + (1-self.beta2)*(grads**2)
        #self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
        #self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
        
        params -= lr_t * self.m / (np.sqrt(self.v_h + 1e-7))
        
        #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
        #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
        #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

def FndVlAccrdIndxLst(AimLst,IndxLst):
    i=0
    RsltLst=[]
    while i<len(IndxLst):
        RsltLst.append(AimLst[IndxLst[i]])
        i=i+1
    return RsltLst

def getLoss(theta):
    mask = np.random.choice(train_size, 1000)
    x = x_train[mask]
    y = t_train[mask]
    loss = np.linalg.norm(x @ theta - y)**2
    return loss / len(mask)
def getGradient(theta, x_batch, y_batch):
    temp = 2 * x_batch.T @ (x_batch @ theta - y_batch)
    temp = temp / x_batch.shape[0]**2
    return temp
def timeCADA(timeList, F):
    if len(timeList) <= F:
        timeList.sort()
        return timeList[-1]
    else:
        timeList.sort()
        return timeList[F-1]
ps_origin = np.random.randn(784)
(x_train, t_train), (x_test, t_test)= load_mnist(flatten=False)
x_train = x_train.reshape((60000, 784))
t_train = x_train @ ps_origin

G = 4
M = 12
M_G = int(M/G)
length = len(x_train)

#划分数据,g代表分组，ng代表不分组
x_train_ng, t_train_ng = [], []
for i in range(M):
    x_train_ng.append(x_train[int(length / M * i) : int(length / M * (i + 1))])
    t_train_ng.append(t_train[int(length / M * i) : int(length / M * (i + 1))])
x_train_g, t_train_g = [], []
for i in range(G):
    x_train_g.append(x_train[int(length / G * i) : int(length / G * (i + 1))])
    t_train_g.append(t_train[int(length / G * i) : int(length / G * (i + 1))])

batch_size = 100
iters_num = 300000
ps1 = np.random.randn(784)    
#%% CADA
ps = copy.deepcopy(ps1)
workers = {}  
train_size = x_train_ng[0].shape[0] - 1
c = 2
train_loss_list_CADA = []
t_CADA = [0]*(iters_num+1)
comm_CADA = [0]*(iters_num+1)
comp_CADA = [0]*(iters_num+1)
miu = 1e-4
optim3 = CADA(0.01)
pool = []
print("CADA:")
for m in range(M):
    workers[m] = copy.deepcopy(ps)
for i in range(iters_num):
    #pool作为放置stale参数的池子
    if i <= 9:
        pool.append(copy.deepcopy(ps))
        dmax = i
    else:
        del pool[0]
        pool.append(copy.deepcopy(ps))
        dmax = 9
    selected = []
    batch_mask = []
    right = 0
    temp_sel = 0
    for d in range(dmax):
        temp_sel += pool[len(pool)-1-d]-pool[len(pool)-d-2]
    if i != 0:
        right = c*np.linalg.norm(temp_sel)**2
    for g in range(M):
        ll = 0
        ll = ps - workers[g]
        left = np.linalg.norm(ll)**2*(1.3**(g-1)+1)**2
        if left >= right:
            selected.append(g)
    for iter1 in range(len(selected)):
        workers[selected[iter1]] = copy.deepcopy(ps)
    loss_CADA = getLoss(ps)
    train_loss_list_CADA.append(loss_CADA)
    print("loss:",loss_CADA, "selected:", selected)
    if loss_CADA <= 0.1:
        break
    temp = 0
    grad_ps = {}
    for g in range(M):
        mask = np.random.choice(x_train_ng[0].shape[0], batch_size)
        x_batch = x_train_ng[g][mask]
        t_batch = t_train_ng[g][mask]
        grad_ps[g] = getGradient(workers[g], x_batch, t_batch)
    for m in range(M):
        temp += grad_ps[m]
    temp = temp / M
    optim3.update(ps, temp)
    if len(selected) != 0:
        t_CADA[i] = t_CADA[i] + np.max(np.random.exponential(miu, [len(selected), 1]))
    comm_CADA[i] = comm_CADA[i] + len(selected)*2
    comp_CADA[i] = comp_CADA[i] + len(selected)
    t_CADA[i+1] = t_CADA[i]
    comm_CADA[i+1] = comm_CADA[i]
    comp_CADA[i+1] = comp_CADA[i]
    

#%% GLAGV
G = 3

#划分数据,g代表分组，ng代表不分组
x_train_g, t_train_g = [], []
for i in range(G):
    x_train_g.append(x_train[int(length / G * i) : int(length / G * (i + 1))])
    t_train_g.append(t_train[int(length / G * i) : int(length / G * (i + 1))])

ps = copy.deepcopy(ps1)
workers = {}  
train_size = x_train_g[0].shape[0] - 1
c = 0.3
train_loss_list_GLAGV = []
t_GLAGV = [0]*(iters_num+1)
comm_GLAGV = [0]*(iters_num+1)
comp_GLAGV = [0]*(iters_num+1)
miu = 1e-4
optim3 = CADA(0.01)
pool = []
print("GLAGV:")
for m in range(G):
    workers[m] = copy.deepcopy(ps)
for i in range(iters_num):
    #pool作为放置stale参数的池子
    if i <= 9:
        pool.append(copy.deepcopy(ps))
        dmax = i
    else:
        del pool[0]
        pool.append(copy.deepcopy(ps))
        dmax = 9
    selected = []
    batch_mask = []
    right = 0
    temp_sel = 0
    for d in range(dmax):
        temp_sel += pool[len(pool)-1-d]-pool[len(pool)-d-2]
    if i != 0:
        right = c*np.linalg.norm(temp_sel)**2
    for g in range(G):
        ll = 0
        ll = ps - workers[g]
        left = np.linalg.norm(ll)**2*(1.3**(g-1)+1)**2
        if left >= right:
            selected.append(g)
    for iter1 in range(len(selected)):
        workers[selected[iter1]] = copy.deepcopy(ps)
    loss_GLAGV = getLoss(ps)
    train_loss_list_GLAGV.append(loss_GLAGV)
    print("loss:",loss_GLAGV, "selected:", selected)
    if loss_GLAGV <= 0.1:
        break
    temp = 0
    grad_ps = {}
    for g in range(G):
        mask = np.random.choice(x_train_g[0].shape[0], batch_size)
        x_batch = x_train_g[g][mask]
        t_batch = t_train_g[g][mask]
        grad_ps[g] = getGradient(workers[g], x_batch, t_batch)
    for m in range(G):
        temp += grad_ps[m]
    temp = temp / G
    optim3.update(ps, temp)
    if len(selected) != 0:
        t_GLAGV[i] = t_GLAGV[i] + np.max(np.min(np.random.exponential(miu, [len(selected), M_G]), axis = 1))
    comm_GLAGV[i] = comm_GLAGV[i] + len(selected)*(1+M_G)
    comp_GLAGV[i] = comp_GLAGV[i] + len(selected)*M_G
    t_GLAGV[i+1] = t_GLAGV[i]
    comm_GLAGV[i+1] = comm_GLAGV[i]
    comp_GLAGV[i+1] = comp_GLAGV[i]
#%% ADAM 
ps = copy.deepcopy(ps1)
workers = {}  
train_size = x_train_ng[0].shape[0] - 1
c = 0
train_loss_list_ADAM = []
t_ADAM = [0]*(iters_num+1)
comm_ADAM = [0]*(iters_num+1)
comp_ADAM = [0]*(iters_num+1)
miu = 1e-4
optim1 = Adam(0.005)
pool = []
print("ADAM:")
for m in range(M):
    workers[m] = copy.deepcopy(ps)
for i in range(iters_num):
    #pool作为放置stale参数的池子
    if i <= 9:
        pool.append(copy.deepcopy(ps))
        dmax = i
    else:
        del pool[0]
        pool.append(copy.deepcopy(ps))
        dmax = 9
    selected = []
    batch_mask = []
    right = 0
    temp_sel = 0
    for d in range(dmax):
        temp_sel += pool[len(pool)-1-d]-pool[len(pool)-d-2]
    if i != 0:
        right = c*np.linalg.norm(temp_sel)**2
    for g in range(M):
        ll = 0
        ll = ps - workers[g]
        left = np.linalg.norm(ll)**2*(1.3**(g-1)+1)**2
        if left >= right:
            selected.append(g)
    for iter1 in range(len(selected)):
        workers[selected[iter1]] = copy.deepcopy(ps)
    loss_ADAM = getLoss(ps)
    train_loss_list_ADAM.append(loss_ADAM)
    print("loss:",loss_ADAM, "selected:", selected)
    if loss_ADAM <= 0.1:
        break
    temp = 0
    grad_ps = {}
    for g in range(M):
        mask = np.random.choice(x_train_ng[0].shape[0], batch_size)
        x_batch = x_train_ng[g][mask]
        t_batch = t_train_ng[g][mask]
        grad_ps[g] = getGradient(workers[g], x_batch, t_batch)
    for m in range(M):
        temp += grad_ps[m]
    temp = temp / M
    optim1.update(ps, temp)
    if len(selected) != 0:
        t_ADAM[i] = t_ADAM[i] + np.max(np.random.exponential(miu, [1, M]))
    comm_ADAM[i] = comm_ADAM[i] + M*2
    comp_ADAM[i] = comp_ADAM[i] + M
    t_ADAM[i+1] = t_ADAM[i]
    comm_ADAM[i+1] = comm_ADAM[i]
    comp_ADAM[i+1] = comp_ADAM[i]
    

#%% LASG
ps = copy.deepcopy(ps1)
workers = {}  
train_size = x_train_ng[0].shape[0] - 1
c = 0
train_loss_list_LASG = []
t_LASG = [0]*(iters_num+1)
comm_LASG = [0]*(iters_num+1)
comp_LASG = [0]*(iters_num+1)
miu = 1e-4
optim3 = SGD(2.6)
pool = []
print("LASG:")
for m in range(M):
    workers[m] = copy.deepcopy(ps)
for i in range(iters_num):
    #pool作为放置stale参数的池子
    if i <= 9:
        pool.append(copy.deepcopy(ps))
        dmax = i
    else:
        del pool[0]
        pool.append(copy.deepcopy(ps))
        dmax = 9
    selected = []
    batch_mask = []
    right = 0
    temp_sel = 0
    for d in range(dmax):
        temp_sel += pool[len(pool)-1-d]-pool[len(pool)-d-2]
    if i != 0:
        right = c*np.linalg.norm(temp_sel)**2
    for g in range(M):
        ll = 0
        ll = ps - workers[g]
        left = np.linalg.norm(ll)**2*(1.3**(g-1)+1)**2
        if left >= right:
            selected.append(g)
    for iter1 in range(len(selected)):
        workers[selected[iter1]] = copy.deepcopy(ps)
    loss_LASG = getLoss(ps)
    train_loss_list_LASG.append(loss_LASG)
    print("loss:",loss_LASG, "selected:", selected)
    if loss_LASG <= 0.1:
        break
    temp = 0
    grad_ps = {}
    for g in range(M):
        mask = np.random.choice(x_train_ng[0].shape[0], batch_size)
        x_batch = x_train_ng[g][mask]
        t_batch = t_train_ng[g][mask]
        grad_ps[g] = getGradient(workers[g], x_batch, t_batch)
    for m in range(M):
        temp += grad_ps[m]
    temp = temp / M
    optim3.update(ps, temp)
    if len(selected) != 0:
        t_LASG[i] = t_LASG[i] + np.max(np.random.exponential(miu, [len(selected), 1]))
    comm_LASG[i] = comm_LASG[i] + len(selected)*2
    comp_LASG[i] = comp_LASG[i] + len(selected)
    t_LASG[i+1] = t_LASG[i]
    comm_LASG[i+1] = comm_LASG[i]
    comp_LASG[i+1] = comp_LASG[i]
    
#%%
mm=5
fs = 13
plt.figure()
plt.semilogy(t_LASG[0:len(train_loss_list_LASG)], train_loss_list_LASG,color='green', linestyle='-')
plt.semilogy(t_ADAM[0:len(train_loss_list_ADAM)], train_loss_list_ADAM,color='blue',linestyle='-', linewidth=3)
plt.semilogy(t_CADA[0:len(train_loss_list_CADA)], train_loss_list_CADA,color='black', linestyle=':')
plt.semilogy(t_GLAGV[0:len(train_loss_list_GLAGV)], train_loss_list_GLAGV,color='red',linestyle='-', linewidth=mm)
#plot(t_GLAGA, train_loss_list_GLAGA)
plt.xlabel('wall-clock time', fontsize=fs)
plt.ylabel('training loss', fontsize=fs)
plt.legend(['D-SGD','D-Adam','CADA','G-CADA'])
plt.grid('on')
#plt.show()
plt.savefig('fig6.pdf', format="pdf")
plt.figure()
plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y',useLocale=(False))
plt.plot(t_LASG[0:len(train_loss_list_LASG)], comm_LASG[0:len(train_loss_list_LASG)],color='green', linestyle='-')
plt.plot(t_ADAM[0:len(train_loss_list_ADAM)], comm_ADAM[0:len(train_loss_list_ADAM)],color='blue',linestyle='-', linewidth=3)
plt.plot(t_CADA[0:len(train_loss_list_CADA)], comm_CADA[0:len(train_loss_list_CADA)],color='black',linestyle=':')
plt.plot(t_GLAGV[0:len(train_loss_list_GLAGV)], comm_GLAGV[0:len(train_loss_list_GLAGV)],color='red',linestyle='-',linewidth=mm)
#plot(t_GLAGVA, comm_GLAGVA)
plt.xlabel('wall-clock time', fontsize=fs) 
plt.ylabel('communication load', fontsize=fs)
plt.legend(['D-SGD','D-Adam','CADA','G-CADA'])
plt.grid('on')
plt.savefig('fig7.pdf', format="pdf")
plt.figure()
plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
plt.plot(t_LASG[0:len(train_loss_list_LASG)], comp_LASG[0:len(train_loss_list_LASG)],color='green', linestyle='-')
plt.plot(t_ADAM[0:len(train_loss_list_ADAM)], comp_ADAM[0:len(train_loss_list_ADAM)],color='blue',linestyle='-', linewidth=3)
plt.plot(t_CADA[0:len(train_loss_list_CADA)], comp_CADA[0:len(train_loss_list_CADA)],color='black',linestyle=':')
plt.plot(t_GLAGV[0:len(train_loss_list_GLAGV)], comp_GLAGV[0:len(train_loss_list_GLAGV)],color='red',linestyle='-',linewidth=mm)
#plot(t_GLAGVA, comp_GLAGVA)
plt.xlabel('wall-clock time', fontsize=fs)
plt.ylabel('computation load', fontsize=fs)
plt.legend(['D-SGD','D-Adam','CADA','G-CADA'])
plt.grid('on')

plt.savefig('fig8.pdf', format="pdf")

'''plt.figure()
x = ['CADA','GAS','ADAM']
y = [len(train_loss_list_CADA), len(train_loss_list_GLAGV), len(train_loss_list_ADAM)]
plt.bar(x, y)
plt.xlabel('methods')
plt.ylabel('# of iterations')
plt.savefig('fig9.pdf', format="pdf")'''




