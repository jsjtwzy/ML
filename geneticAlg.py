# -*-coding:utf-8 -*-
#目标求解2*sin(x)+cos(x)最大值
import random
import math
import matplotlib.pyplot as plt
import time
import numpy as np
import geneticAlgOpt
class GA(object):
#初始化种群 生成chromosome_length大小的population_size个个体的种群
 
    def __init__(self,population_size,chromosome_length,max_value,pc,pm, epochs):
 
        self.population_size=population_size
        self.choromosome_length=chromosome_length
        # self.population=[[]]
        self.max_value=max_value
        self.pc=pc
        self.pm=pm
        self.epochs = epochs
        # self.fitness_value=[]
 
 
 
    def species_origin(self):
        population=[]
        for _ in range(self.population_size):
 
            temporary = np.random.randint(0, 2, (self.choromosome_length))
            population.append(temporary)
            #将染色体添加到种群中
        return np.array(population)
            # 将种群返回，种群是个二维数组，个体和染色体两维
 
    #从二进制到十进制
    #编码  input:种群,染色体长度 编码过程就是将多元函数转化成一元函数的过程
    def translation(self,population):
        
        #make binary2decimal weights
        weights = (2 *np.ones((self.choromosome_length))) **np.arange((self.choromosome_length)) /(math.pow(2,self.choromosome_length)-1)
        temporary = population @weights.T
        
        #一个染色体编码完成，由一个二进制数编码为一个十进制数
        return temporary
   # 返回种群中所有个体编码完成后的十进制数
 
 
 
#from protein to function,according to its functoin value
 
#a protein realize its function according its structure
# 目标函数相当于环境 对染色体进行筛选，这里是2*sin(x)+math.cos(x)
    def function(self,population):
        
        temporary=self.max_value *self.translation(population)
        f = lambda x :2*np.sin(x)+np.cos(x)
        function1 = f(temporary)
 
         #这里将sin(x)作为目标函数
        return function1
 
#定义适应度
    def fitness(self,function1):
 
        fitness_value=np.zeros(np.shape(function1))
        fitness_value[function1 > 0] = function1[function1 > 0]
 
        #only function_value > 0 will be counted
        #将适应度添加到列表中
 
        return fitness_value
 
#计算适应度和采用内置sum
 
 
#计算适应度斐伯纳且列表
    def cumsum(self,fitness1):
        fitness1 = np.cumsum(fitness1)
 
 
#3.选择种群中个体适应度最大的个体
    def selection(self,population,fitness_value):
        
    #单个公式暂存器
        total_fitness=sum(fitness_value)
    #将所有的适应度求和
        new_fitness = fitness_value /total_fitness
    #将所有个体的适应度正则化
        self.cumsum(new_fitness)
    
    #存活的种群
        population_length=pop_len=len(population)
    #求出种群长度
    #根据随机数确定哪几个能存活
        ms=np.random.rand(pop_len)
    # 产生种群个数的随机值
    # ms.sort()
    # 存活的种群排序
        fitin=0
        newin=0
        new_population=new_pop=population
 
    #轮盘赌方式
        while newin<pop_len and fitin < pop_len:
              if ms[newin]<new_fitness[fitin]:
                new_pop[newin]=population[fitin]
                newin+=1
              else:
                fitin+=1
        population=new_pop
 
#4.交叉操作
    def crossover(self,population):
#pc是概率阈值，选择单点交叉还是多点交叉，生成新的交叉个体，这里没用
        pop_len=len(population)
 
        for i in range(pop_len-1):
 
            if(random.random()<self.pc):
 
               cpoint=random.randint(0,len(population[0]))
           #在种群个数内随机生成单点交叉点
               temporary1 = np.copy(population[i, 0:cpoint])
 
               population[i, 0:cpoint]=population[i+1, 0:cpoint]
               population[i+1, 0:cpoint]=temporary1
        # 第i个染色体和第i+1个染色体基因重组/交叉完成
    def mutation(self,population):
     # pm是概率阈值
         px=len(population)
    # 求出种群中所有种群/个体的个数
         py=len(population[0])
    # 染色体/个体基因的个数
         for i in range(px):
             if(random.random()<self.pm):
                mpoint=random.randint(0,py-1)
            #
                population[i][mpoint] = 1 - population[i][mpoint]
 
#transform the binary to decimalism
# 将每一个染色体都转化成十进制 max_value,再筛去过大的值
    def b2d(self,best_individual):
        
        weights = (2 *np.ones((self.choromosome_length))) **np.arange((self.choromosome_length))
        total = best_individual @weights.T
 
        total=total*self.max_value/(math.pow(2,self.choromosome_length)-1)
        return total
 
#寻找最好的适应度和个体
 
    def best(self,population,fitness_value):
 
        px=len(population)
        bestfitness=max(fitness_value)
        bestindividual=population[fitness_value == bestfitness][0]
        # print(fitness_value)
 
        return [bestindividual,bestfitness]
 
 
    def plot(self, results, Y2):
        X = []
        Y = []
 
        for i in range(self.epochs):
            X.append(i)
            Y.append(results[i][0])
 
        plt.plot(X, Y, Y2)
        plt.show()
 
    def main(self):
 
        start = time.perf_counter()
        results = [[]]
        fitness_value = []
        fitmean = []
 
        population = pop = self.species_origin()
 
        for i in range(self.epochs):
            function_value = self.function(population)
            # print('fit funtion_value:',function_value)
            fitness_value = self.fitness(function_value)
            # print('fitness_value:',fitness_value)
 
            best_individual, best_fitness = self.best(population,fitness_value)
            results.append([best_fitness, self.b2d(best_individual)])
        # 将最好的个体和最好的适应度保存，并将最好的个体转成十进制,适应度函数
            self.selection(population,fitness_value)
            self.crossover(population)
            self.mutation(population)
        results = results[1:]
        results.sort()
        results2 = geneticAlgOpt.gaOpt()
        print(time.perf_counter()-start)
        self.plot(results, results2)
 
if __name__ == '__main__':
 
 
   population_size=400
   max_value=2
   chromosome_length=20
   pc=0.6
   pm=0.01
   ga=GA(population_size,chromosome_length,max_value,pc,pm, 50)
   ga.main()