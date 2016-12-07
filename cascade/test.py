from statsmodels.base.model import GenericLikelihoodModel
import random


endog = [2,3,4,1,2,1,2,3,4,1,2,3,4,5,1,2,3,4,5,2,1,1,3,2,2,2,3,4,5,6,7,8,2,4,7,8]

import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize



def main():
    global endog
    p = 0.1
    q = 0.2
    exog = []
    for each_m in endog:
        exog.append(get_informed(each_m, p, q))





def get_informed(m, p, q):
    """

    :param m: no of active neighbors
    :param p: innovation param
    :param q: immitation param
    :return:
    """
    p = 1 - (1 - p) * pow((1 - q), m)
    u = random.uniform(0, 1)
    if u < p:
        return 1
    return 0




class myProbit(GenericLikelihoodModel):
    def loglike(self, params):
        exog = self.exog
        endog = self.endog
        p = params[0]
        q = params[1]
        r = 1 - (1 - p) * pow((1 - q), endog)
        return stats.norm.logcdf(r*np.dot(exog, params)).sum()





if __name__== "__main__":
    main()



