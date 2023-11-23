import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

train_df = pd.read_csv('train.csv')

input_cols = ['GrLivArea', 'YearBuilt']
target = 'SalePrice'


kf = KFold(n_splits=5, random_state=42, shuffle=True)

metrics = {'linear_reg':[],
           'svr':[],
           'dt':[],
           'rf':[]}

models = [LinearRegression(), DecisionTreeRegressor(),
          SVR(), RandomForestRegressor()]
model_names = ['linear_reg', 'svr', 'dt', 'rf']

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df.index)):

    train_set = train_df.loc[train_idx]
    val_set = train_df.loc[val_idx]

    for name, model in zip(model_names, models):
        reg = model.fit(train_set[input_cols], train_set[target])
        pred = reg.predict(val_set[input_cols])
        score = mean_squared_error(val_set[target], pred)
        metrics[name].append(score)

#### result ####
for name in metrics.keys():
    print(f'Model {name}:', np.round(np.mean(metrics[name]), 3))
