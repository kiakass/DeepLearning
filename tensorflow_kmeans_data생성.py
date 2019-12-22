# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 14:07:12 2019

@author: Administrator

k-means를 위한 그루핑 데이터 생성, 그래프 그리기
"""

import tensorflow as tf
import pandas as pd   # 데이터조작
import seaborn as sns # 시각화패키지
import matplotlib.pyplot as plt
import numpy as np

num_points = 40
vectors_set,vectors1_set,vectors2_set,vectors3_set =[], [], [], []

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors1_set.append([np.random.normal(0.0,0.9),
                            np.random.normal(0.0,0.9)])
    else:
        vectors2_set.append([np.random.normal(3.0,0.5),
                            np.random.normal(1.0,0.5)])
print(len(vectors1_set))
df = pd.DataFrame({"x": [v[0] for v in vectors1_set],
                   "y": [v[1] for v in vectors1_set]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=6)

print(len(vectors2_set))
df = pd.DataFrame({"x": [v[0] for v in vectors2_set],
                   "y": [v[1] for v in vectors2_set]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=6)

vectors3_set = vectors1_set + vectors2_set

print(len(vectors3_set))

df = pd.DataFrame({"x": [v[0] for v in vectors3_set],
                   "y": [v[1] for v in vectors3_set]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=3)
plt.show()    

'''
matlab 그래프 그리기 sample
'''
x = np.arange(1,10)
y = x*5
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('matplotlib sample')
plt.plot(x,y,'or')
plt.show()
