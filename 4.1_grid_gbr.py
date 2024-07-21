import pandas as pd
import numpy as np
from function import process_df
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('data/nofill.csv')
data = data.drop(['PW', 'Molarity', 'Cation', 'Anion'], axis=1)
data_encoded = pd.get_dummies(data, columns=['Morphology'], drop_first=False)
X_train, X_test, y_train, y_test = process_df(data_encoded)

# 初始化 GradientBoostingRegressor 模型
gbr = GradientBoostingRegressor(n_estimators=100, random_state=21)

# 定义超参数网格，每隔一定距离搜索
param_grid = {
    'learning_rate': np.arange(0.16, 0.26, 0.02),
    'max_depth': np.arange(5, 10, 1),
    'min_samples_leaf': np.arange(1, 2, 1),
    'min_samples_split': np.arange(3, 6, 1),
    'subsample': np.arange(0.1, 1.1, 0.1),
    'max_features': np.arange(0.1, 1.1, 0.1)
}

# 使用 GridSearchCV 进行超参数搜索
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best Parameters:")
print(grid_search.best_params_)
