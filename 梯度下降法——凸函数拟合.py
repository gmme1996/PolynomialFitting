# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class gradient:
    def __init__(self, times):
        self.times = times
        self.x = np.arange(0, np.pi*2, np.pi*2/self.times)        
        self.y = [np.sin(xi) for xi in self.x]
        self.seita = np.array([0.,0.,0.,0.])
        self.step = 0.001
        self.matching()
    def matching(self):
        store_seita = []
        store_seita.append(self.seita)
        k = 1
        while k < 20:
            temp_seita = np.array([0.,0.,0.,0.])
            for j in range(self.times):
                temp_h = 0
                for i in range(len(self.seita)):
                    temp_h += self.seita[i]*np.power(self.x[j], i)
                temp_y = self.y[j]
                temp_x = self.x[j]
                for i in range(len(temp_seita)):
                    temp_seita[i] += (temp_h - temp_y)*np.power(temp_x, i)
            while True:
                print (self.seita)
                temp = self.seita
                cost_first = 0
                for i in range(self.times):
                    temp_h = 0
                    for j in range(len(self.seita)):
                        temp_h += self.seita[j]*np.power(self.x[i], j)
                    temp_y = self.y[i]
                    cost_first += (temp_h - temp_y)*(temp_h - temp_y)/2.0
                self.seita = self.seita - temp_seita*self.step/self.times
                cost_second = 0
                for i in range(self.times):
                    temp_h = 0
                    for j in range(len(self.seita)):
                        temp_h += self.seita[j]*np.power(self.x[i], j)
                    temp_y = self.y[i]
                    cost_second += (temp_h - temp_y)*(temp_h - temp_y)/2.0
                if cost_second - cost_first > 0:
                    self.seita = temp
                    self.step -= 0.0001
                else:
                    break
            store_seita.append(self.seita)
            k += 1
            delta = 0
            for i in range(len(self.seita)):
                delta += (store_seita[0][i]-self.seita[i])*(store_seita[0][i]-self.seita[i])
            delta=np.sqrt(delta)
        while delta > 0.0001:
            temp_seita = np.array([0.,0.,0.,0.])
            for j in range(self.times):
                temp_h = 0
                for i in range(len(self.seita)):
                    temp_h += self.seita[i]*np.power(self.x[j], i)
                temp_y = self.y[j]
                temp_x = self.x[j]
                for i in range(len(temp_seita)):
                    temp_seita[i] += (temp_h - temp_y)*np.power(temp_x, i)
            while True:
                print (self.seita)
                temp = self.seita
                cost_first = 0
                for i in range(self.times):
                    temp_h = 0
                    for j in range(len(self.seita)):
                        temp_h += self.seita[j]*np.power(self.x[i], j)
                    temp_y = self.y[i]
                    cost_first += (temp_h - temp_y)*(temp_h - temp_y)/2.0
                self.seita = self.seita - temp_seita*self.step/self.times
                cost_second = 0
                for i in range(self.times):
                    temp_h = 0
                    for j in range(len(self.seita)):
                        temp_h += self.seita[j]*np.power(self.x[i], j)
                    temp_y = self.y[i]
                    cost_second += (temp_h - temp_y)*(temp_h - temp_y)/2.0
                if cost_second - cost_first > 0:
                    self.seita = temp
                    self.step -= 0.0001
                else:
                    break
            for i in range(1,20):
                store_seita[i-1] = store_seita[i]
            store_seita[19] = self.seita
            delta = 0
            for i in range(len(self.seita)):
                delta += (store_seita[0][i]-self.seita[i])*(store_seita[0][i]-self.seita[i])
            delta=np.sqrt(delta)
        h_label = []
        for i in range(self.times):
            temp=0
            for j in range(len(self.seita)):
                temp += self.seita[j]*np.power(self.x[i], j)
            h_label.append(temp)
        plt.plot(self.x,self.y, 'g')
        plt.axis([0.,2*np.pi,-1.5,1.5])
        plt.plot(self.x, h_label, 'r')
        plt.show()
test = gradient(100)
