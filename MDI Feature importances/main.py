from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

california = fetch_california_housing()

X = california["data"]
Y = california["target"]
names = california["feature_names"]

rf = RandomForestRegressor()
rf.fit(X, Y)

forest_importances = pd.Series(rf.feature_importances_, index=california.feature_names)

std = np.std([rf.feature_importances_ for tree in rf.estimators_], axis=0)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title('')
fig.tight_layout()
