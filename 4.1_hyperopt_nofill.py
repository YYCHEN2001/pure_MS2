import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split
from hyperopt.pyll.base import scope
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import numpy as np
from sklearn.preprocessing import StandardScaler

# Step 1: Define the search space
space = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'gamma': hp.uniform('gamma', 0, 1),
    'max_depth': scope.int(hp.quniform('max_depth', 3, 18, 1)),
    'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 9, 1)),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
    'colsample_bynode': hp.uniform('colsample_bynode', 0.5, 1),
}

df_nofill = pd.read_csv('data/nofill.csv')


# Step 2: Objective function
def objective(params):
    params['max_depth'] = int(params['max_depth'])
    params['min_child_weight'] = int(params['min_child_weight'])
    clf = xgb.XGBRegressor(**params)
    score = -cross_val_score(clf, X, y, scoring='neg_root_mean_squared_error', cv=5).mean()
    return {'loss': score, 'status': STATUS_OK}

# Assuming X and y are your features and target variable
# For demonstration, let's split df_nofill into X and y
X = df_nofill.drop('Cs', axis=1)
y = df_nofill['Cs']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=21)

# Step 3: Run the optimization
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=200,  # You can adjust this number to your needs
            trials=trials)

print("Best hyperparameters:", best)