import pandas as pd
import numpy as np
import statsmodels.formula.api as stats

x = [1,2,3,4,5]
y = [8,7,6,4,5]

a = np.array(71,58,92,78,71,68,67,88,88,60,80,70,68,82,78) # 강사 1
b = np.array(50,65,75,91,67,39,81,68,97,86,66,60,65,55,58) # 강사 2

statistics, p = stats.ttest_1samp(a,b)
print('검정 통계량 : ', statistics)
print('p-value : ', p)