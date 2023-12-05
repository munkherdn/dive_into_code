import gc
import joblib
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

gc.enable()

### Data load ####
DEBUG = False
REMOVE_OUTLIERS=True

if DEBUG:
    train_df = pd.read_csv('data/application_train.csv', nrows=1000)
    test_df = pd.read_csv('data/application_test.csv', nrows=1000)
else:
    train_df = pd.read_csv('data/application_train.csv')
    test_df = pd.read_csv('data/application_test.csv')

train_df['isTrain'] = 'Train'
test_df['isTrain'] = 'Test'

test_df['TARGET'] = np.nan

df = pd.concat([train_df, test_df[train_df.columns]], axis=0)
df = df.reset_index(drop=True)

del train_df, test_df
gc.collect()

### Data preprocessing ###
num_cols = []
cat_cols = []
should_be_encode = []
not_useful_cols = ['SK_ID_CURR', 'TARGET', 'isTrain']

for col in df.columns:
    if col not in not_useful_cols:
        unique_len = len(df[col].unique())
        data_type = df[col].dtype

        if unique_len<=20 and data_type!="object":
            cat_cols.append(col)
        elif data_type=='object':
            should_be_encode.append(col)
        else:
            num_cols.append(col)

print('Number of cat cols:', len(cat_cols+should_be_encode))
print('Number of numerical cols:', len(num_cols))

### Fill missing values ###
missing_values = {}
for col in num_cols:
    num_missing_values = df[col].isnull().sum()
    if num_missing_values>0:
        df[col] = df[col].fillna(np.nanmean(df[col].values))
        missing_values[col] = np.nanmean(df[col].values)

for col in should_be_encode:
    num_missing_values = df[col].isnull().sum()
    if num_missing_values > 0:
        df[col] = df[col].fillna(df[col].mode())
        missing_values[col] = (df[col].mode())

for col in cat_cols:
    num_missing_values = df[col].isnull().sum()
    if num_missing_values > 0:
        #df[col] = df[col].fillna(df[col].mode())
        df[col] = df[col].fillna(df[col].median())
        missing_values[col] = (df[col].median())

joblib.dump(missing_values, 'utils/missing_values.pkl')

#### label encoding ####
encoders = {}
for col in should_be_encode:
    encoder = LabelEncoder().fit(df[col])
    encoders[col] = encoder
    df[col] = encoder.transform(df[col])

joblib.dump(encoders, "utils/encoders.pkl",)



### verify ####
for col in cat_cols+should_be_encode+num_cols:
    if df[col].isnull().sum()>0:
        print(col, df[col].dtype, df[col].isnull().sum())

cat_cols = cat_cols + should_be_encode
del should_be_encode
gc.collect()


#### onehot encoding ####
dummy_cols = []
map_dummy = {}
for col in tqdm(cat_cols, total=len(cat_cols)):
    for value in tqdm(df[col].unique()):
        df[f'dummy_{col}_{value}'] = 0
        df.loc[df[col]==value, f'dummy_{col}_{value}'] = 1
        dummy_cols.append(f'dummy_{col}_{value}')
        map_dummy[col] = f"dummy_{col}_{value}"

joblib.dump(map_dummy, 'utils/map_dummy.pkl')

del cat_cols
gc.collect()

### outlier remove ####
train_df = df.loc[df['isTrain']=='Train'].reset_index(drop=True)
test_df = df.loc[df['isTrain']=='Test'].reset_index(drop=True)

del df
gc.collect()

outlier_idx = []
for col in num_cols:
    z_score = stats.zscore(train_df[col])
    outlier_idx = outlier_idx + list(train_df.loc[(np.abs(z_score)>3)].index)

outlier_idx = list(set(outlier_idx))

if REMOVE_OUTLIERS:
    train_df = train_df.drop(index=outlier_idx,
                             axis=0).reset_index(drop=True)

joblib.dump(num_cols, 'utils/num_cols.pkl')
joblib.dump(dummy_cols, 'utils/dummy_cols.pkl')

### Modelling ###
skf = StratifiedKFold(n_splits=5,
                      shuffle=True,
                      random_state=42)

models = []
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df,
                                    train_df['TARGET'],
                                    groups=train_df['TARGET'])):
    train_set = train_df.loc[train_idx]
    val_set = train_df.loc[val_idx]

    model = LogisticRegression().fit(train_set[num_cols+dummy_cols],
                                     train_set['TARGET'])
    models.append(model)

    y_pred = model.predict_proba(val_set[num_cols+dummy_cols])[:, 1]

    auc_score = roc_auc_score(val_set['TARGET'], y_pred)
    print(f"FOLD-{fold}: AUC score={np.round(auc_score, 3)}")

joblib.dump(models, 'utils/models.pkl')

