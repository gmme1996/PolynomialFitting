# -*- coding: utf-8 -*-
import math
import random
import numpy
import scipy as sp
import matplotlib.pyplot as plt
#常量定义区
W=[]
trainnum=100
#拟合函数次数
times=9
tail=times+1
X = numpy.arange(-1,1,2.0/trainnum)
Y = [((xi-1)*(xi-1)+0.5)*numpy.cos(xi) for xi in X]
plt.plot(X,Y,'.') 
step=0.0001
xlabel=[]
ylabel=[]
select={}
for i in range(tail):
	W.append(1);
#拟合函数初始化
for i in range(trainnum/10):
	xlabel.append(random.uniform(-1,1))
	ylabel.append(Y[i])
 #开始迭代
for lamata in range(1): 
	while True:
		cost_first=0
           for i in range(tail):
               cost_first+=W[i]**2
	      cost_first=lamata*math.sqrt(cost_first)
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
	          t=lamata*W[i]/temp
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
		     cost_second+=W[i]**2
	      cost_second=lamata*math.sqrt(cost_second)
	      for j in range(trainnum):
		      temp=0
		      for q in range(tail):
			      temp+=W[q]*(X[j]**q)
			      if q==0:
				      temp-=Y[j]
		cost_second+=temp**2/(2*trainnum)
		if cost_first-cost_second<10e-10:
			break
	f = sp.poly1d(W)
	fx=sp.linspace(-1,1,100)
	er=sp.sum((f(xlabel)-ylabel)**2)
	plt.plot(fx, f(fx)+0.5)
	select[lamata]=[]
	select[lamata].append(er)
	select[lamata].append(list(W))
	for i in range(tail):
		W[i]=1   
for lamata in range(1):
	r=0
	if select[lamata][0] < select[r][0]:
		r=lamata   
plt.show()

