# -*- coding: utf-8 -*-
import random
import numpy
import scipy as sp
import matplotlib.pyplot as plt
#常量定义区
W=[]
times=9
tail=times+1
trainnum=100
step=0.00001
xlabel=[]
ylabel=[]
select={}
#数据随机生成
X = numpy.arange(-1,1,2.0/trainnum)
Y = [((xi-1)*(xi*xi-1)+0.5)*numpy.sin(xi) for xi in X]
plt.plot(X,Y,'r^')    
Sigma=0.05
for i in range(tail):
	W.append(1);
#你和函数初始化
for i in range(trainnum/10):
	xlabel.append(random.uniform(-1,1))
	ylabel.append(Y[i])
#迭代
for l in range(3): 
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
           print (cost_first-cost_second)
           if cost_first-cost_second<10e-10:
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
for l in range(3):
      r=0
      if select[l][0] < select[r][0]:
            r=l
plt.show()