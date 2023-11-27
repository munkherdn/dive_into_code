import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('train.csv')

input_cols = ['GrLivArea', 'YearBuilt']
target = 'SalePrice'
train_df[target] = np.log(train_df[target])

train_set, test_set = train_test_split(train_df, test_size=0.2,
                                       shuffle=True, random_state=42)

metrics = {'linear_reg':[],
           'svr':[],
           'dt':[]}

models = [LinearRegression(), DecisionTreeRegressor(),
          SVR()]
model_names = ['linear_reg', 'svr', 'dt',]
preds = {}
oofs = []
for name, model in zip(model_names, models):
    reg = model.fit(train_set[input_cols], train_set[target])
    oofs.append(reg.predict(train_set[input_cols]))
    pred = reg.predict(test_set[input_cols])
    score = mean_squared_error(test_set[target], pred)
    metrics[name].append(score)
    preds[name] = pred

#### result ####
for name in metrics.keys():
    print(f'Model {name}:', np.round(np.mean(metrics[name]), 3))

### blending ####
weights = [0.4, 0.2, 0.4]

final_pred = None
for i, name in enumerate(model_names):
    if i==0:
        final_pred = weights[i]*preds[name]
    else:
        final_pred = final_pred + weights[i]*preds[name]

score = mean_squared_error(test_set[target], final_pred)
print(f'Blending:', np.round(np.mean(score), 3))

### stacking ####
stacking_model = LinearRegression()
X_pred = np.asarray(oofs).T
X_test_pred = np.zeros((test_set.shape[0], 3))
for i, name in enumerate(model_names):
    X_test_pred[:, i] = preds[name]

reg = stacking_model.fit(X_pred, train_set[target])
pred_stack = reg.predict(X_test_pred)

score = mean_squared_error(test_set[target], pred_stack)
print(f'Stacking:', np.round(np.mean(score), 3))

#### bagging ###
bagging = 5

metrics = {'linear_reg': [],
           'svr': [],
           'dt': []}

models = [LinearRegression(), DecisionTreeRegressor(),
          SVR()]
model_names = ['linear_reg', 'svr', 'dt', ]
preds = {}

for i in range(bagging):
    frac = np.random.randint(80, 90)
    train_ = train_set.sample(int(train_set.shape[0]*frac/100))
    for name, model in zip(model_names, models):
        reg = model.fit(train_[input_cols], train_[target])
        pred = reg.predict(test_set[input_cols])
        score = mean_squared_error(test_set[target], pred)
        metrics[name].append(score)
        if i==0:
            preds[name] = pred/bagging
        else:
            preds[name] += pred / bagging

#### result ####
for name in metrics.keys():
    print(f'Model {name}:', np.round(np.mean(metrics[name]), 3))

weights = [0.4, 0.2, 0.4]

final_pred = None
for i, name in enumerate(model_names):
    if i==0:
        final_pred = weights[i]*preds[name]
    else:
        final_pred = final_pred + weights[i]*preds[name]

score = mean_squared_error(test_set[target], final_pred)
print(f'Bagging and Blending:', np.round(np.mean(score), 3))
