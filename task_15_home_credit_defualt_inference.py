import joblib
from tqdm import tqdm
import pandas as pd
import numpy as np


test_df = pd.read_csv('data/application_test.csv')
test_df = test_df.iloc[:1]

#missing values
missing_values = joblib.load('utils/missing_values.pkl')
for col in missing_values.keys():
    test_df.loc[test_df[col].isnull(), col] = missing_values[col]

## encoding ###
encoders = joblib.load('utils/encoders.pkl')
for col in encoders.keys():
    test_df[col] = encoders[col].transform(test_df[col])

### dummmy
cat_cols = ['CNT_CHILDREN', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_QRT', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
for col in tqdm(cat_cols, total=len(cat_cols)):
    for value in tqdm(test_df[col].unique()):
        test_df[f'dummy_{col}_{value}'] = 0
        test_df.loc[test_df[col]==value, f'dummy_{col}_{value}'] = 1

### prediction ###
models = joblib.load('utils/models.pkl')
num_cols = joblib.load('utils/num_cols.pkl')
dummy_cols = joblib.load('utils/dummy_cols.pkl')

for col in dummy_cols:
    if col not in test_df.columns:
        test_df[col] = 0

y_pred = np.zeros(test_df.shape[0])
for model in models:
    y_pred += model.predict_proba(test_df[num_cols+dummy_cols])[:, 1]/len(models)

test_df['TARGET'] = y_pred
test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False)
