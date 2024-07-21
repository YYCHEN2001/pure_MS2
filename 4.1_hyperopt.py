import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from hyperopt.pyll.base import scope
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from function import process_df

space = {
    'learning_rate': hp.quniform('learning_rate', 0.01, 0.3, 0.01),
    'subsample': hp.quniform('subsample', 0, 1, 0.05),
    'gamma': hp.quniform('gamma', 0, 1, 0.05),
    'max_depth': scope.int(hp.quniform('max_depth', 3, 20, 1)),
    'min_child_weight': hp.quniform('min_child_weight', 0, 20, 0.05),
    'reg_alpha': hp.quniform('reg_alpha', 0, 2, 0.05),
    'reg_lambda': hp.quniform('reg_lambda', 0, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'colsample_bylevel': hp.quniform('colsample_bylevel', 0.5, 1, 0.05),
    'colsample_bynode': hp.quniform('colsample_bynode', 0.5, 1, 0.05),
    'random_state': 42
}

data = pd.read_csv('data/nofill.csv')
data = data.drop(['PW'], axis=1)
# data_encoded = pd.get_dummies(data, columns=['Morphology'], drop_first=False)


def objective(params):
    params['max_depth'] = int(params['max_depth'])
    clf = xgb.XGBRegressor(**params)
    score = -cross_val_score(clf, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5).mean()
    return {'loss': score, 'status': STATUS_OK}


# 划分训练集和测试集
data['target_class'] = pd.qcut(data['Cs'], q=10, labels=False)
X_train, X_test, y_train, y_test = process_df(data)

# Step 3: Run the optimization
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=200,  # You can adjust this number to your needs
            trials=trials)

best_hyperparameters = {
    'learning_rate': float(best['learning_rate']),  # Adjusting scale if necessary
    'subsample': float(best['subsample']),
    'gamma': float(best['gamma']),
    'max_depth': int(best['max_depth']),
    'min_child_weight': float(best['min_child_weight']),
    'reg_alpha': float(best['reg_alpha']),
    'reg_lambda': float(best['reg_lambda']),
    'colsample_bytree': float(best['colsample_bytree']),
    'colsample_bylevel': float(best['colsample_bylevel']),
    'colsample_bynode': float(best['colsample_bynode']),
}

# Convert the dictionary to a DataFrame for a tabular display
df_best_hyperparameters = pd.DataFrame([best_hyperparameters])

# Adjust display settings
pd.set_option('display.max_columns', None)  # Ensure all columns are displayed
pd.set_option('display.width', None)        # Use maximum width available
# Display the DataFrame
print(df_best_hyperparameters)
