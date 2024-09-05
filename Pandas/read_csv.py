import numpy as np
import pandas as pd

df = pd.read_csv('UNSW_NB15_training-set.csv', header=None)

print(df[3])

df = df[3]

df.to_csv('vuluu.csv')