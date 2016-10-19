import numpy 
import scipy as sp  
import pylab as pl  
from scipy.optimize import leastsq   
#常量声明区
times = 15
tail=times+1   
regularization = 0.1
x = numpy.arange(-2,2,0.1)
y = [((xi-1)*(xi-1)+0.5)*numpy.cos(xi) for xi in x] 
ylabel = [np.random.normal(0, 0.1) + y for y in y]  # 添加正太分布噪声后的函数  
# 多项式函数  
def fit(p, x):  
    f = numpy.poly1d(p)  
    return f(x)  
# 残差函数  
def residuals(p, y, x):  
    f=numpy.poly1d(p)
    ret=f(x)-y
    ret = np.append(ret, np.sqrt(regularization) * p)   # 将lambda^(1/2)p加在了返回的array的后面  
    return ret     
tempx = np.linspace(-2, 2, 1000)   
  
p= np.random.randn(tail)  # 随机初始化多项式参数  
plsq = leastsq(residuals, p, args=(ylabel, x))  
print '拟合函数系数: ', plsq[0]  # 输出拟合参数  
pl.plot(tempx, fit(plsq[0], tempx))  
pl.plot(x, ylabel, 'go')  
