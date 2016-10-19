# -*- coding: utf-8 -*-
"""
Spyder Editor
Created on Wed Oct 15 16:05:28 2016
@author: 匡盟盟
1：最小二乘法
2：加惩罚项最小二乘法
3：梯度下降法求正弦函数
4：梯度下降法（可加惩罚项）
5：共轭梯度法（可加惩罚项）
"""
print(__doc__)
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy as sp
from scipy.optimize import leastsq
f = plt.figure()
draw = f.add_subplot(111)
#常量定义区
xlabel=[]
ylabel=[]
W=[]
A=[]
B=[]
X=[]
Y=[]
tempA=[]
tempB=[]
tail=0
step=0.001
#生成数据
def createdata(trainnum,times):
    X = np.arange(-1,1,2.0/trainnum)
    Y = [((x-1)*(x*x-1)+0.5)*np.sin(x) for x in X]
    plt.plot(X,Y,'r^')
    tail=times+1
#产生噪点
def producenoise():
    i = 0
    for x in X:
          r=float(random.randint(80,100))/100
          xlabel.append(x*r)
          ylabel.append(Y[i]*r)
          i+=1
#最小二乘法
def lsm(trainnum,times):
    createdata(trainnum,times)
    producemoise()
    for i in range(tail):
	    tempA=[]
	    for j in range(tail):
		    temp=0.0
		    for k in range(trainnum):
			    d=1.0
			    for m in range(0,j+i):
				    d*=xlabel[k]
			    temp+=d
		    tempA.append(temp)
         A.append(tempA)
     for i in range(tail):
	     temp=0.0
	     for k in range(trainnum):
		     d=1.0
		     for j in range(i):
			    d=d*xlabel[k]
                temp+=ylabel[k]*d
	     B.append(temp)
      #求解AXY=B,即在文档中的最后一部分
     XY=np.linalg.solve(A,B)
     print('系数的矩阵表示为：[a0,---,a%d]'%(times))
     print(XY)
     for i in range(0,length):
	     temp=0.0
	     for j in range(0,tail):
		     d=1.0
		     for k in range(0,j):
			     d*=XY[i]
		     d*=XY[j]
		     temp+=d
	     Y.append(temp)
     draw.plot(X,Y,color='r',linestyle='-',marker='.')
#加惩罚项的最小二乘法
def rlsm(trainnum,times,lamata=1):  
    createdata(trainnum,times)
    producenoise() 
    def fit(p, xlabel):  
        f = numpy.poly1d(p)  
        return f(xlabel)  
    #残差函数  
    def residuals(p, ylabel, xlabel):  
        f=numpy.poly1d(p)
        ret=f(xlabel)-ylabel
        ret = np.append(ret, np.sqrt(lamata/10.0) * p)   
        return ret     
    xlabel = np.linspace(-2, 2, 1000)   
    p= np.random.randn(tail) 
    #调用第三方库
    plsq = leastsq(residuals, p, args=(ylabel, xlabel))  
    print ('系数的矩阵表示: ', plsq[0]) 
    pl.plot(xlabel, fit(plsq[0], xlabel))  
    pl.plot(xlabel, ylabel, 'go')  
#梯度下降拟合正弦曲线
def gds(trainnum,times):
    createdata(trainnum,times)
    producenoise()
#因为是正弦曲线，需要重新生成Y
    Y = [np.sin(x) for x in X]
    seita = np.array([0.,0.,0.,0.])
    store_seita = []
    store_seita.append(seita)
    k = 1
#控制循环次数
    while k < 20:
        temp_s = np.array([0.,0.,0.,0.])
        for j in range(trainnum):
            temp_h = 0
            for i in range(len(seita)):
                temp_h += seita[i]*np.power(X[j], i)
            tempA = Y[j]
            tempB= X[j]
            for i in range(len(temp_s)):
                temp_s[i] += (temp_h - tempA)*np.power(tempB, i)
#只要前后两次代价查不满足要求，就一直迭代
        while True:
            temp = seita
            cost_first = 0
            for i in range(trainnum):
                 temp_h = 0
                 for j in range(len(seita)):
                      temp_h += seita[j]*np.power(X[i], j)
                  tempA = Y[i]
                  cost_first += (temp_h - tempA)*(temp_h - tempA)/2.0
            seita-=temp_s*step/trainnum
            cost_second = 0
            for i in range(trainnum):
                temp_h = 0
                for j in range(len(seita)):
                     temp_h += seita[j]*np.power(X[i], j)
                tempA = Y[i]
                cost_second += (temp_h - tempA)*(temp_h - tempA)/2.0
            if cost_second - cost_first > 0:
                seita = temp
                step -= 0.0001
            else:
                break
        store_seita.append(seita)
        k += 1
        delta = 0
        for i in range(len(seita)):
            delta += (store_seita[0][i]-seita[i])*(store_seita[0][i]-seita[i])
        delta=np.sqrt(delta)
#用栈来保存新生成的seita
        for i in range(10):
            store_seita[i-1] = store_seita[i]
        store_seita[9] = seita
        delta = 0
        for i in range(len(seita)):
            delta += (store_seita[0][i]-seita[i])*(store_seita[0][i]-seita[i])
        delta=np.sqrt(delta)
    ylabel = []
    for i in range(trainnum):
          temp=0
          for j in range(len(seita)):
              temp += seita[j]*np.power(X[i], j)
          ylabel.append(temp)
    plt.plot(X, ylabel, 'r')
    plt.show()
#梯度下降法（lamata为||w||2的系数）
def gd(trainnum,times,lamata=1):
    createdata(trainnum,times)
    producenoise()
    select={}
#初始设定一个w
    for i in range(tail):
	    W.append(0.1);
    for l in range(lamata): 
#前后代价差小于设定时，才结束
          while True:
		    cost_first=0
               for i in range(tail):
                     cost_first+=W[i]**2
	         cost_first=l*math.sqrt(cost_first)
	         for j in range(trainnum):
		         temp=0
		         for q in range(tail):
			         temp+=W[q]*(X[j]**q)
			         if q==0:
				        temp-=Y[j]
		    cost_first+=temp**2/(2*trainnum)
		    for i in range(tail):
                     temp=0
	               for q in range(tail):
		               temp+=W[q]**2
	                temp=math.sqrt(temp)
                 #求w导数
	                t=l*W[i]/temp
	                for j in range(trainnum):
		                temp=0
		                for q in range(tail):
			                temp+=W[q]*(X[j]**q)
			                if q==0:
				               temp-=Y[j]
		                t+=temp*i*W[i]*X[j]**(i-1)/trainnum	
			     W[i]-=step*t
		   cost_second=0
              for i in range(tail):
		        cost_second+=W[i]**2
	        cost_second=l*math.sqrt(cost_second)
	        for j in range(trainnum):
		        temp=0
		        for q in range(tail):
			        temp+=W[q]*(X[j]**q)
			        if q==0:
				        temp-=Y[j]
		  cost_second+=temp**2/(2*trainnum)
		  if cost_first-cost_second<10e-5:
			   break
          f = sp.poly1d(W)
          fx=sp.linspace(-1,1,100)
          er=sp.sum((f(xlabel)-ylabel)**2)
          plt.plot(fx, f(fx)+0.5)
          select[l]=[]
          select[l].append(er)
          select[l].append(list(W))
          for i in range(tail):
		    W[i]=0.1   
    for l in range(1):
	   r=0
	   if select[l][0] < select[r][0]:
		   r=l   
    plt.show()
#共轭梯度法
def cg(trainnum,times,lamata=1):
    createdata(trainnum,times)
    producenoise()
#初始生成w
    for i in range(tail):
	    W.append(0.1);
    for l in range(lamata): 
#迭代代价差小于设定时退出
	    while True:
               cost_first=0
               for i in range(tail):
                    cost_first+=l*(W[i]**2)
               for j in range(trainnum):
                    temp=0
                    for q in range(tail):
                        temp+=W[q]*(X[j]**q)
                        if q==0:
                             temp-=Y[j]
                    cost_first+=temp**2/(2*trainnum)
               for i in range(tail):
                   #求导数
                    t=2*l*W[i]
                    for j in range(trainnum):
                         temp=0
                         for q in range(tail):
                              temp+=W[q]*(X[j]**q)
                              if q==0:
                                  temp-=Y[j]
                    t+=temp*i*W[i]*X[j]**(i-1)/trainnum	
                    W[i]=W[i]-step*t
                cost_second=0
                for i in range(tail):
                     cost_second+=l*(W[i]**2)
                for j in range(trainnum):
                     temp=0
                     for q in range(tail):
                          temp+=W[q]*(X[j]**q)
                          if q==0:
                               temp-=Y[j]
                     cost_second+=temp**2/(2*trainnum)
                if cost_first-cost_second<10e-5:
                     break
	    f = sp.poly1d(W)
	    fx=sp.linspace(-1,1,100)
          e=sp.sum(f(x)-y)**2)
	    plt.plot(fx, f1(fx)-1)
	    select[l]=[]
	    select[l].append(er)
	    select[l].append(list(W))
	    for i in range(tail):
		    W[i]=1
    for l in range(lamata):
         r=0
         if select[l][0] < select[r][0]:
               r=l
    plt.show()
#实际测试  
select = input("请输入所要使用的拟合函数:")
trainnum=input("请输入拟合节点数量：")
times=input("请输入拟合曲线次数：")
lamata=input("请输入加惩罚项时的lamata（整数，默认为1）：")
if(select==1):
    lsm(trainnum,times)
elif(select==2):
    rlsm(trainnum,times,lamata)
elif(select==3):
    gds(trainnum,times)
elif(select==4):
    gd(trainnum,times,lamata)
elif(select==5):
    cg(trainnum,times,lamata)
else:
    print("暂未收录此算法！\n")
