# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 14:07:12 2019

@author: Administrator
"""
import tensorflow as tf
import pandas as pd   # 데이터조작
import seaborn as sns # 시각화패키지
import matplotlib.pyplot as plt
import numpy as np

num_points = 2000
vectors_set,vectors1_set,vectors2_set =[], [], []

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors1_set.append([np.random.normal(0.0,0.9),
                            np.random.normal(0.0,0.9)])
    else:
        vectors2_set.append([np.random.normal(3.0,0.5),
                            np.random.normal(1.0,0.5)])
print(vectors1_set)
df = pd.DataFrame({"x": [v[0] for v in vectors1_set],
                   "y": [v[1] for v in vectors1_set]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=6)

df = pd.DataFrame({"x": [v[0] for v in vectors2_set],
                   "y": [v[1] for v in vectors2_set]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=6)

vectors3_set = vectors1_set.append(vectors2_set)

plt.show()    


