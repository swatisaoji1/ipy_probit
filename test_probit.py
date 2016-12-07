from __future__ import print_function
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel


'''
The data listed in Appendix Table F21.1 were taken from a study by Spector and Mazzeo
(1980), which examined whether a new method of teaching economics, the Personalized
System of Instruction (PSI), significantly influenced performance in later economics courses.
The "dependent variable" used in our application is GRADE, which indicates the whether
a student's grade in an intermediate macroeconomics course was higher than that in the
principles course. The other variables are GPA, their grade point average; TUCE, the score
on a pretest that indicates entering knowledge of the material; and PSI, the binary variable
indicator of whether the student was exposed to the new teaching method. (Spector and
Mazzeo's specific equation was somewhat different from the one estimated here.)

'''
data = sm.datasets.spector.load_pandas()


print("DATASET DESCRIPTION: ------>")
print(sm.datasets.spector.NOTE)

# exog = regressors i.e independent variables here
#  In Our case Regressor/Dependent variable is Adopted/NonAdopted outcome
print("EXOG VARIABLES (INDEPENDENT VARS: ------>")
exog = data.exog
print(data.exog.head())

# endog = Dependent Variables He
print("ENDOG VARIABLES (DEPENDENT VARS/ REGRESSORS): ------>")
endog = data.endog
print(data.endog.head())



print("STEPS ---->")
print("STEP 1. add contant column to EXOG")
exog = sm.add_constant(exog, prepend=True)
print(exog[:5])


print("STEP 2. Define the Likehood Function")
class MyProbit(GenericLikelihoodModel):
    def loglike(self, params):
        exog = self.exog
        endog = self.endog
        q = 2 * endog - 1
        return stats.norm.logcdf(q*np.dot(exog, params)).sum()


print("STEP 3. Fit the Probit Model based on your own likelihood function")
sm_probit_manual = MyProbit(endog, exog).fit()

print("STEP 4. Print the Summary of the fit")
print(sm_probit_manual.summary())

