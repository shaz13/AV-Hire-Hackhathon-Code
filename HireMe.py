print ("Starting ML Engines...")
import pandas as pd
print ("Pandas version", pd.__version__)
import numpy as np
print ("Numpy version", np.__version__)
import sklearn as sk
print ("Sci-kit learn version", sk.__version__)
import lightgbm as lgb
print ("LightGBM learn version", lgb.__version__)
import xgboost as xgb
print ("XGBoost learn version", xgb.__version__)
print ("ML Engines are loaded..")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor


# Functions for Daytimes and Is Weekend Feature
def DayTimes(Hour):
    if Hour in range(0,5):
        return "MidNight"
    elif Hour in range(5,7):
        return "Morning"
    elif Hour in range(7,17):
        return "Afternoon"
    elif Hour in range(17,24):
        return "Night"

def Weekend(DOW):
    if DOW in [0,5,6]:
        return 1
    else:
        return 0

def RMSE(y_actual, preds, log = False):
    if log:
        RMSE = sqrt(mean_squared_error(y_actual, np.expm1(preds)))
    else:
        RMSE = sqrt(mean_squared_error(y_actual, preds))
    return RMSE

def LOCAL_TEST():
    n = 10
    YEAR_LIST = [2013, 2014, 2015, 2016, 2017]
    for year in YEAR_LIST:
        print ("Predicting for the year...", year)
        Month_RMSE = []
        for month in set(X_TRAIN_RAW['Month']):
            YR = year
            if year == 2013:
                M = 7
                while M <= 12:
                    X_train = X_TRAIN_RAW[(X_TRAIN_RAW['Day'] <= n) & (X_TRAIN_RAW['Month'] <= M) & (X_TRAIN_RAW['Year'] <= YR) ].drop('electricity_consumption', axis=1)
                    y_train = X_TRAIN_RAW[(X_TRAIN_RAW['Day'] <= n) & (X_TRAIN_RAW['Month'] <= M) & (X_TRAIN_RAW['Year'] <= YR) ].electricity_consumption

                    X_test = X_TRAIN_RAW[(X_TRAIN_RAW['Day'] > n) & (X_TRAIN_RAW['Month'] <= M) & (X_TRAIN_RAW['Year'] <= YR) ].drop('electricity_consumption', axis=1)
                    y_test = X_TRAIN_RAW[(X_TRAIN_RAW['Day'] > n) & (X_TRAIN_RAW['Month'] <= M) & (X_TRAIN_RAW['Year'] <= YR) ].electricity_consumption


                    X_train = X_train[cols]

                    light.fit(X_train, np.log1p(y_train))
                    pred = light.predict(X_test[cols])

                    Month_RMSE.append(RMSE(y_test, pred, log=True))
                    M = M + 1
            else:

                M= month

                X_train = X_TRAIN_RAW[(X_TRAIN_RAW['Day'] <= n) & (X_TRAIN_RAW['Month'] <= M) & (X_TRAIN_RAW['Year'] <= YR) ].drop('electricity_consumption', axis=1)
                y_train = X_TRAIN_RAW[(X_TRAIN_RAW['Day'] <= n) & (X_TRAIN_RAW['Month'] <= M) & (X_TRAIN_RAW['Year'] <= YR) ].electricity_consumption

                X_test = X_TRAIN_RAW[(X_TRAIN_RAW['Day'] > n) & (X_TRAIN_RAW['Month'] <= M) & (X_TRAIN_RAW['Year'] <= YR) ].drop('electricity_consumption', axis=1)
                y_test = X_TRAIN_RAW[(X_TRAIN_RAW['Day'] > n) & (X_TRAIN_RAW['Month'] <= M) & (X_TRAIN_RAW['Year'] <= YR) ].electricity_consumption

                X_train = X_train[cols]

                light.fit(X_train, np.log1p(y_train))
                pred = light.predict(X_test[cols])

                Month_RMSE.append(RMSE(y_test, pred, log=True))
        print (np.mean(Month_RMSE))

def TIME_MACHINE():
    pd.options.mode.chained_assignment = None  # default='warn'
    n = 23
    GENERATED_TEST  = pd.DataFrame()
    YEAR_LIST = [2013, 2014, 2015, 2016, 2017]
    for year in YEAR_LIST:
        print ("Predicting for the year:", year)
        for month in (set(X_TRAIN_RAW['Month'])):
            YR = year

            if year == 2013:
                M = 7
                while M <= 12:
                    X  = X_TRAIN_RAW[(X_TRAIN_RAW['Day'] <= n) & (X_TRAIN_RAW['Month'] <= M) & (X_TRAIN_RAW['Year'] <= YR) ].drop('electricity_consumption', axis=1)
                    y = X_TRAIN_RAW[(X_TRAIN_RAW['Day'] <= n) & (X_TRAIN_RAW['Month'] <= M) & (X_TRAIN_RAW['Year'] <= YR) ].electricity_consumption
                    X_TEST  = X_TEST_RAW[(X_TEST_RAW['Day'] > 23 ) & (X_TEST_RAW['Month'] <= M) & (X_TEST_RAW['Year'] <= YR) ].drop('electricity_consumption', axis=1)

                    X = X[cols]
                    light.fit(X, np.log1p(y))
                    pred = light.predict(X_TEST[cols])
                    X_TEST['electricity_consumption'] = pred
                    GENERATED_TEST = pd.concat([GENERATED_TEST,X_TEST])
                    M = M + 1
            else:
                M = month
                X = X_TRAIN_RAW[(X_TRAIN_RAW['Day'] < n) & (X_TRAIN_RAW['Month'] <= M) & (X_TRAIN_RAW['Year'] <= YR) ].drop('electricity_consumption', axis=1)
                y = X_TRAIN_RAW[(X_TRAIN_RAW['Day'] < n) & (X_TRAIN_RAW['Month'] <= M) & (X_TRAIN_RAW['Year'] <= YR) ].electricity_consumption
                X_TEST  = X_TEST_RAW[(X_TEST_RAW['Day'] >= 23 ) & (X_TEST_RAW['Month'] <= M) & (X_TEST_RAW['Year'] <= YR) ].drop('electricity_consumption', axis=1)
                X = X[cols]
                light.fit(X, np.log1p(y))
                pred = light.predict(X_TEST[cols])
                X_TEST['electricity_consumption'] = pred
                GENERATED_TEST = pd.concat([GENERATED_TEST,X_TEST])
    GENt = GENERATED_TEST[['Year', 'Day','Month', 'Hour','DOW','electricity_consumption']]
    test_feats['ID'] = test['ID'].values
    REAL_PREDS = pd.merge(GENt,test_feats , on=['Year','Day', 'Month', 'Hour', 'DOW'])
    REAL_PREDS = REAL_PREDS[['electricity_consumption_x','ID']].sort_values(by='ID', ascending=1).rename(columns={
    'electricity_consumption_x':'electricity_consumption'})[['ID','electricity_consumption']]
    PREDS =  np.expm1(REAL_PREDS['electricity_consumption']).values
    REAL_PREDS['electricity_consumption'] = PREDS
    return PREDS,REAL_PREDS

print ("Importing the data")
train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
sample_sub = pd.read_csv('Data/sample_submission_q0Q3I1Z.csv')

# Concating Train and Test Sets
alldata = pd.concat([train,test])
alldata['datetime'] = pd.to_datetime(alldata['datetime'])

print ("Data succesfully loaded")

print ("Feature Enginnering: Adding Datetime properties (1/3)")
alldata['Day'] = alldata['datetime'].dt.day
alldata['DOW'] = alldata['datetime'].dt.dayofweek
alldata['Year'] = alldata['datetime'].dt.year
alldata['Month'] = alldata['datetime'].dt.month
alldata['Hour'] = alldata['datetime'].dt.hour

print ("Feature Enginnering: Groupby Mean, Min, Sum and Max of the factor variables (2/3)")
temp = alldata.groupby(['DOW','Hour'], as_index=False)['electricity_consumption'].max().rename(columns={'electricity_consumption':'max_electricity_consumption'})
alldata = pd.merge(alldata, temp, how='left', on=['DOW','Hour'])
temp = alldata.groupby(['Hour','DOW'], as_index=False)['pressure'].min().rename(columns={'pressure':'min_pressure'})
alldata = pd.merge(alldata, temp, how='left', on=['DOW','Hour'])
temp = alldata.groupby(['DOW','Hour'], as_index=False)['pressure'].mean().rename(columns={'pressure':'mean_pressure'})
alldata = pd.merge(alldata, temp, how='left', on=['DOW','Hour'])
temp = alldata.groupby(['DOW','Hour'], as_index=False)['pressure'].max().rename(columns={'pressure':'max_pressure'})
alldata = pd.merge(alldata, temp, how='left', on=['DOW','Hour'])
temp = alldata.groupby(['Hour','DOW'], as_index=False)['windspeed'].min().rename(columns={'windspeed':'min_windspeed'})
alldata = pd.merge(alldata, temp, how='left', on=['DOW','Hour'])
temp = alldata.groupby(['DOW','Hour'], as_index=False)['windspeed'].mean().rename(columns={'windspeed':'mean_windspeed'})
alldata = pd.merge(alldata, temp, how='left', on=['DOW','Hour'])
temp = alldata.groupby(['DOW','Hour'], as_index=False)['windspeed'].max().rename(columns={'windspeed':'max_windspeed'})
alldata = pd.merge(alldata, temp, how='left', on=['DOW','Hour'])
temp = alldata.groupby(['Hour','DOW'], as_index=False)['temperature'].min().rename(columns={'temperature':'min_temperature'})
alldata = pd.merge(alldata, temp, how='left', on=['DOW','Hour'])
temp = alldata.groupby(['DOW','Hour'], as_index=False)['temperature'].mean().rename(columns={'temperature':'mean_temperature'})
alldata = pd.merge(alldata, temp, how='left', on=['DOW','Hour'])
temp = alldata.groupby(['DOW','Hour'], as_index=False)['temperature'].max().rename(columns={'temperature':'max_temperature'})
alldata = pd.merge(alldata, temp, how='left', on=['DOW','Hour'])
temp = alldata.groupby(['Hour','DOW'], as_index=False)['var1'].min().rename(columns={'var1':'min_var1_temperature'})
alldata = pd.merge(alldata, temp, how='left', on=['DOW','Hour'])
temp = alldata.groupby(['DOW','Hour'], as_index=False)['var1'].mean().rename(columns={'var1':'mean_var1_temperature'})
alldata = pd.merge(alldata, temp, how='left', on=['DOW','Hour'])
temp = alldata.groupby(['DOW','Hour'], as_index=False)['var1'].max().rename(columns={'var1':'max_var1_temperature'})
alldata = pd.merge(alldata, temp, how='left', on=['DOW','Hour'])
del temp

print ("Feature Enginnering: Time of the Day, Weekend Proximity, Holiday Flag, Relations with var1 (3/3)")
alldata['DayTime'] = alldata['Hour'].apply(DayTimes)
alldata['WeekendFlag'] = alldata['DOW'].apply(Weekend)
alldata['Holiday'] = alldata.DOW > 5

alldata['VAR_TEMP'] = alldata['var1']/ alldata['temperature']
alldata['VAR_PRE'] = alldata['var1'] / alldata['pressure']
alldata['VAR_WIN'] = alldata['var1']/ alldata['windspeed']

alldata['INV_VAR_TEMP'] = alldata['temperature'] /alldata['var1']
alldata['INV_VAR_PRE'] =  alldata['pressure']/alldata['var1']
alldata['INV_VAR_WIN'] = alldata['windspeed']/alldata['var1']
alldata.drop('var1', axis=1, inplace=True)

encoder = LabelEncoder()
alldata['var2'] = encoder.fit_transform(alldata['var2'])
alldata['DayTime'] = encoder.fit_transform(alldata['DayTime'])

cols_to_drop = ['datetime', 'ID']
alldata.drop(cols_to_drop, axis=1,inplace=True)

train_feats = alldata[~pd.isnull(alldata.electricity_consumption)]
test_feats = alldata[pd.isnull(alldata.electricity_consumption)]

X_TEST_RAW = test_feats.copy()
X_TRAIN_RAW = train_feats.copy()

cols = ['pressure','temperature','var2','windspeed','Day','DOW','Year','Month','Hour','max_electricity_consumption','min_pressure',
        'mean_pressure','max_pressure',  'min_windspeed','mean_windspeed', 'max_windspeed','min_temperature', 'mean_temperature', 'max_temperature',
        'min_var1_temperature', 'mean_var1_temperature', 'max_var1_temperature','DayTime',  'Holiday', 'VAR_TEMP', 'VAR_PRE', 'VAR_WIN',
        'INV_VAR_TEMP', 'INV_VAR_PRE', 'INV_VAR_WIN']

print ("Testing on the train set for decreased RMSE error as the data is cumilatively increasing")
light = XGBRegressor()
LOCAL_TEST()
print ("Done. Positive results!")

print ("Cooking the Submission file...")
pred, sub = TIME_MACHINE()
sub['electricity_consumption'] = pred
sub.to_csv('Shaz13_Submission.csv', index=None)
print ("Exported the submission")
print ("Done...")
