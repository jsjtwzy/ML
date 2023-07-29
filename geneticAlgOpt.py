from sko.GA import GA
import numpy as np
import matplotlib.pyplot as plt
import math

class TagetFunc():
    def __init__(self, chrom_len, max_value=1, min_value=0):
        self.max_value = max_value
        self.min_value = min_value
        self.chrom_len = chrom_len

    def translation(self, p):
        
        #make binary2decimal weights
        weights = self.max_value *(2 *np.ones((len(p)))) **np.arange((len(p))) /(math.pow(2,len(p))-1)
        temporary = (p >0.5).astype(int) @weights.T
        
        #一个染色体编码完成，由一个二进制数编码为一个十进制数
        return temporary

    def function(self, p):
        x = self.translation(p)
        result = 2 *np.sin(x) +np.cos(x)
        return result

def gaOpt():
    target = TagetFunc(20, max_value=5)
    chrom_len = target.chrom_len
    func = target.function
    # ndim can't be 1 ,so that crossover vill not be done
    ga = GA(func=func, n_dim=chrom_len, size_pop=40, max_iter=50, prob_mut=0.01, ub=1, lb=0)
    ga.run()
    his_y = np.array(ga.all_history_FitV)
    max_y = np.max(his_y,1)
    best_x = target.translation(np.array(ga.best_x))
    print(best_x)
    
    return max_y
    
if __name__ == '__main__':
    max_y = gaOpt()
    plot1 = plt.plot(max_y)
    plt.show()