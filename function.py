import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import train_test_split


# 定义一个函数来处理DataFrame,包括分层抽样
def process_df(df):
    df['target_class'] = pd.qcut(df['Cs'], q=10, labels=False)
    X = df.drop(['Cs', 'target_class'], axis=1)
    y = df['Cs']
    stratify_column = df['target_class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_column)
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred):
    """
    Calculate and return actual vs pred fig for data_dopants metrics.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = root_mean_squared_error(y_true, y_pred)
    return r2, mae, mape, rmse


def metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, model_name):
    R2_train, MAE_train, MAPE_train, RMSE_train = calculate_metrics(y_train, y_train_pred)
    R2_test, MAE_test, MAPE_test, RMSE_test = calculate_metrics(y_test, y_test_pred)
    metrics = {'model': model_name,
               'R2_train': R2_train, 'MAE_train': MAE_train, 'MAPE_train': MAPE_train, 'RMSE_train': RMSE_train,
               'R2_test': R2_test, 'MAE_test': MAE_test, 'MAPE_test': MAPE_test, 'RMSE_test': RMSE_test}
    model_name_df = pd.DataFrame([metrics])
    return model_name_df


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot_actual_vs_predicted(y_train, y_pred_train, y_test, y_pred_test, figtitle, figpath=None):
    """
    Plot the actual vs predicted values for both training and test sets,
    and plot y=x as the fit line.
    """
    # 设置全局字体为Times New Roman，字号为32，字体粗细为粗体
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 24,
        'font.weight': 'bold',
        'figure.figsize': (6, 6)  # 设置图像尺寸为6x6英寸
    })

    # 在绘制之前设置背景颜色
    plt.gcf().set_facecolor('white')
    plt.gca().set_facecolor('white')
    
    # 绘制训练集和测试集的散点图
    plt.scatter(y_train, y_pred_train, color='blue', label='Train', s=50, alpha=0.5)
    plt.scatter(y_test, y_pred_test, color='red', label='Test', s=50, alpha=0.5)

    # 计算合并数据的最小值和最大值，用于设置坐标轴范围和绘制 y=x 线
    y_pred_train = y_pred_train.ravel()
    y_pred_test = y_pred_test.ravel()
    y_combined = np.concatenate([y_train, y_pred_train, y_test, y_pred_test])

    # 动态获取数据的最小值和最大值
    min_val, max_val = np.min(y_combined), np.max(y_combined)

    # 为 y=x 线留出额外的间距，以适应数据的范围
    padding = (max_val - min_val) * 0.1
    padded_min, padded_max = min_val - padding, max_val + padding

    # 将 padded_min 和 padded_max 调整为最近的整百值
    padded_min = np.floor(padded_min / 100) * 100
    padded_max = np.ceil(padded_max / 100) * 100

    # 绘制 y=x 的虚线，线宽为3
    plt.plot([padded_min, padded_max], [padded_min, padded_max], 'k--', lw=3, label='Regression Line')

    # 设置标题和轴标签
    plt.title(figtitle, fontweight='bold', pad=20, color='black')
    plt.xlabel('Actual Values', fontweight='bold', color='black')
    plt.ylabel('Predicted Values', fontweight='bold', color='black')

    # 设置图例，无边框，位于左上角，文字为黑色
    legend = plt.legend(frameon=False, loc='upper left', fontsize=16)
    for text in legend.get_texts():
        text.set_color('black')

    # 设置刻度线的长度和粗细
    plt.tick_params(axis='both', which='major', length=10, width=2, labelsize=20, labelcolor='black', color='black')

    # 设置 X 和 Y 轴的刻度，确保刻度为整百
    x_ticks = np.arange(padded_min, padded_max + 100, 400)
    y_ticks = np.arange(padded_min, padded_max + 100, 400)

    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    # 设置坐标轴范围为 padded_min 到 padded_max，确保 y=x 线覆盖整个数据范围
    plt.xlim([padded_min, padded_max])
    plt.ylim([padded_min, padded_max])

    # 设置图形边界的宽度和可见性
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_color('black')

    # 保存图像，背景透明，紧凑布局
    plt.savefig(figpath, bbox_inches='tight', transparent=True)
    plt.show()
