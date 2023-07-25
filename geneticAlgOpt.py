from sko.GA import GA
import numpy as np
import matplotlib.pyplot as plt
import math


def translation(p):
    
    max_value = 10
    #make binary2decimal weights
    weights = max_value *(2 *np.ones((len(p)))) **np.arange((len(p))) /(math.pow(2,len(p))-1)
    temporary = p @weights.T
        
    #一个染色体编码完成，由一个二进制数编码为一个十进制数
    return temporary
def function(p):
    x = translation(p)
    print(x)
    return 2 *np.sin(x) +np.cos(x)


chromosome_length=20
# ndim can't be 1 ,so that crossover vill not be done
ga = GA(func=function, n_dim=chromosome_length, size_pop=40, max_iter=50, prob_mut=0.01, ub=1, lb=0)
ga.run()
his_y = np.array(ga.all_history_FitV)
max_y = np.max(his_y,1)
plot1 = plt.plot(max_y)
plt.show()