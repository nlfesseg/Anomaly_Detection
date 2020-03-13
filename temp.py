import pandas as pd
from matplotlib import pyplot
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("out.csv")
data = data.drop(columns='DateTime')
drop_cols = data.columns[(data == 0).sum() > 0.9 * data.shape[0]]
data.drop(drop_cols, axis=1, inplace=True)

# len = len(data.columns)
# values = data.values
# X = values[:, 0:len]
# Y = values[:, 2]
# # Create correlation matrix
# fs = SelectKBest(score_func=f_regression, k=5)
# # apply feature selection
# X_selected = fs.fit_transform(X, Y)
#
# corr_matrix = data.corr().abs()
# target = corr_matrix.iloc[1, :]
# result = target[target > 0.8]
#
# print(X_selected.shape)

array = data.values
X = array[:, 0:-1]
y = array[:, -1]
# perform feature selection
rfe = RFE(RandomForestRegressor(n_estimators=1, random_state=1), 4)
fit = rfe.fit(X, y)
# report selected features
print('Selected Features:')
names = data.columns.values[0:-1]
for i in range(len(fit.support_)):
    if fit.support_[i]:
        print(names[i])
# plot feature rank
names = data.columns.values[0:-1]
ticks = [i for i in range(len(names))]
pyplot.bar(ticks, fit.ranking_)
pyplot.xticks(ticks, names)
pyplot.show()
