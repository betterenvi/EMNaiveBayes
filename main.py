import sys, os, collections, copy
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from EMNaiveBayes import *

data_fn = 'data/Soybean/soybean-large.data.txt'
# read data with dtype = str
X = pd.read_csv(data_fn, sep=',', header=None, dtype=str, skiprows=None, na_values='?', keep_default_na=False)
X.dropna(inplace=True)
col_idx_of_Y = 0
Y = X[col_idx_of_Y]
K = len(set(Y))
X.drop(col_idx_of_Y, axis=1, inplace=True)

emnb = EMNaiveBayes(epsilon=1e-5)
emnb.fit(X.values, K)
emnb.evaluate(Y)
