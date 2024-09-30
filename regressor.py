import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, cohen_kappa_score,
                             confusion_matrix, f1_score, accuracy_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# 回帰モデルをインポート
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

# ベイズ最適化用のライブラリ
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# ブートストラップ用
from sklearn.utils import resample

# statsmodelsをインポート
import statsmodels.api as sm
from scipy.stats import norm

# データの読み込み
train = pd.read_csv('train.csv')

# 目的変数と説明変数の定義
X = train.drop(['Id', 'Response'], axis=1)
y = train['Response']

# カテゴリカルと数値データの分離
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 前処理の定義
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの定義
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(random_state=42),
    'Ridge': Ridge(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42)
}

# グリッドサーチのパラメータ
param_grids = {
    'Linear Regression': {},
    'Lasso': {'model__alpha': [0.001, 0.01, 0.1, 1]},
    'Ridge': {'model__alpha': [0.001, 0.01, 0.1, 1]},
    'Decision Tree': {'model__max_depth': [5, 10, 15, None]},
    'Random Forest': {'model__n_estimators': [100, 200], 'model__max_depth': [10, 20, None]},
    'LightGBM': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1]}
}

results = []

for name, model in models.items():
    print(f"Training {name}...")
    # パイプラインの作成
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
    
    # グリッドサーチの実行
    grid_search = GridSearchCV(clf, param_grids.get(name, {}), cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # テストデータでの予測
    y_pred = best_model.predict(X_test)
    
    # モデルごとの予測分散の推定
    if name == 'Linear Regression':
        # statsmodelsで予測の分散を取得
        X_train_sm = sm.add_constant(preprocessor.fit_transform(X_train))
        ols = sm.OLS(y_train, X_train_sm).fit()
        X_test_sm = sm.add_constant(preprocessor.transform(X_test))
        predictions = ols.get_prediction(X_test_sm)
        pred_summary = predictions.summary_frame(alpha=0.05)
        y_pred_var = pred_summary['mean_se'] ** 2  # 標準誤差の二乗が分散
        y_pred = pred_summary['mean']
    elif name == 'Lasso':
        # ブートストラップによる予測分散の推定
        n_bootstraps = 100
        y_preds_bootstrap = np.zeros((n_bootstraps, len(X_test)))
        for i in range(n_bootstraps):
            X_sample, y_sample = resample(X_train, y_train)
            clf_boot = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('model', Lasso(alpha=best_model.named_steps['model'].alpha, random_state=42))])
            clf_boot.fit(X_sample, y_sample)
            y_preds_bootstrap[i] = clf_boot.predict(X_test)
        y_pred = np.mean(y_preds_bootstrap, axis=0)
        y_pred_var = np.var(y_preds_bootstrap, axis=0)
    elif name == 'Random Forest':
        # 各決定木の予測の分散を取得
        model = best_model.named_steps['model']
        all_predictions = np.array([tree.predict(best_model.named_steps['preprocessor'].transform(X_test)) for tree in model.estimators_])
        y_pred = np.mean(all_predictions, axis=0)
        y_pred_var = np.var(all_predictions, axis=0)
    elif name == 'LightGBM':
        # クォンタイル回帰で予測区間を取得し、分散を近似
        # 下位予測（5%点）
        params_lower = {
            'objective': 'quantile',
            'alpha': 0.05,
            'verbose': -1
        }
        model_lower = LGBMRegressor(**params_lower, random_state=42)
        model_lower.fit(preprocessor.fit_transform(X_train), y_train)
        y_pred_lower = model_lower.predict(preprocessor.transform(X_test))
        
        # 上位予測（95%点）
        params_upper = {
            'objective': 'quantile',
            'alpha': 0.95,
            'verbose': -1
        }
        model_upper = LGBMRegressor(**params_upper, random_state=42)
        model_upper.fit(preprocessor.fit_transform(X_train), y_train)
        y_pred_upper = model_upper.predict(preprocessor.transform(X_test))
        
        # 中央予測
        y_pred = best_model.predict(X_test)
        y_pred_var = ((y_pred_upper - y_pred_lower) / 4) ** 2  # 標準偏差を近似し、二乗して分散に
    else:
        # その他のモデル（分散を平均二乗誤差で近似）
        residuals = y_test - y_pred
        y_pred_var = np.full_like(y_pred, fill_value=np.var(residuals))
    
    # 閾値の最適化（ベイズ最適化）
    from skopt.space import Space
    from skopt.space import Real
    from skopt.utils import use_named_args
    
    # 予測値の範囲を取得
    min_pred = y_pred.min()
    max_pred = y_pred.max()
    
    # 最適化するパラメータ空間を定義
    space = [Real(min_pred, max_pred, name=f't{i}') for i in range(7)]
    
    @use_named_args(space)
    def objective(**thresholds):
        # 閾値をソート
        thresh = np.array([thresholds[f't{i}'] for i in range(7)])
        thresh.sort()
        
        # 閾値に基づいてクラスに割り当て
        y_pred_class = np.digitize(y_pred, bins=thresh) + 1  # クラスは1から8
        
        # Quadratic Weighted Kappaの計算
        kappa = cohen_kappa_score(y_test, y_pred_class, weights='quadratic')
        return -kappa  # 最小化するので負にする
    
    # ベイズ最適化の実行
    res = gp_minimize(objective, space, n_calls=50, random_state=42)
    
    # 最適な閾値を取得
    optimal_thresholds = np.array(res.x)
    optimal_thresholds.sort()
    
    # 最適な閾値でクラスに割り当て
    y_pred_class = np.digitize(y_pred, bins=optimal_thresholds) + 1  # クラスは1から8
    
    # 評価指標の計算
    mae = mean_absolute_error(y_test, y_pred_class)
    mse = mean_squared_error(y_test, y_pred_class)
    kappa = cohen_kappa_score(y_test, y_pred_class, weights='quadratic')
    accuracy = accuracy_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class, average='weighted')
    precision = precision_score(y_test, y_pred_class, average='weighted')
    recall = recall_score(y_test, y_pred_class, average='weighted')
    
    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Quadratic Weighted Kappa': kappa
    })

# 結果の表示
results_df = pd.DataFrame(results)
print("\nModel Comparison Results:")
print(results_df.to_string(index=False))
results_df.to_csv('model_comparison_results.csv', index=False)

# 最良のモデルを選択（Quadratic Weighted Kappaが最大のモデル）
best_model_name = results_df.loc[results_df['Quadratic Weighted Kappa'].idxmax(), 'Model']
print(f"\nBest Model: {best_model_name}")

# 最良モデルでの詳細な評価
best_model = models[best_model_name]
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', best_model)])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# モデルごとの予測分散の推定（最良モデル）
if best_model_name == 'Linear Regression':
    X_train_sm = sm.add_constant(preprocessor.fit_transform(X_train))
    ols = sm.OLS(y_train, X_train_sm).fit()
    X_test_sm = sm.add_constant(preprocessor.transform(X_test))
    predictions = ols.get_prediction(X_test_sm)
    pred_summary = predictions.summary_frame(alpha=0.05)
    y_pred_var = pred_summary['mean_se'] ** 2
    y_pred = pred_summary['mean']
elif best_model_name == 'Lasso':
    n_bootstraps = 100
    y_preds_bootstrap = np.zeros((n_bootstraps, len(X_test)))
    for i in range(n_bootstraps):
        X_sample, y_sample = resample(X_train, y_train)
        clf_boot = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', Lasso(alpha=clf.named_steps['model'].alpha, random_state=42))])
        clf_boot.fit(X_sample, y_sample)
        y_preds_bootstrap[i] = clf_boot.predict(X_test)
    y_pred = np.mean(y_preds_bootstrap, axis=0)
    y_pred_var = np.var(y_preds_bootstrap, axis=0)
elif best_model_name == 'Random Forest':
    model = clf.named_steps['model']
    all_predictions = np.array([tree.predict(clf.named_steps['preprocessor'].transform(X_test)) for tree in model.estimators_])
    y_pred = np.mean(all_predictions, axis=0)
    y_pred_var = np.var(all_predictions, axis=0)
elif best_model_name == 'LightGBM':
    # クォンタイル回帰で予測区間を取得し、分散を近似
    params_lower = {
        'objective': 'quantile',
        'alpha': 0.05,
        'verbose': -1
    }
    model_lower = LGBMRegressor(**params_lower, random_state=42)
    model_lower.fit(preprocessor.fit_transform(X_train), y_train)
    y_pred_lower = model_lower.predict(preprocessor.transform(X_test))
    
    params_upper = {
        'objective': 'quantile',
        'alpha': 0.95,
        'verbose': -1
    }
    model_upper = LGBMRegressor(**params_upper, random_state=42)
    model_upper.fit(preprocessor.fit_transform(X_train), y_train)
    y_pred_upper = model_upper.predict(preprocessor.transform(X_test))
    
    y_pred = clf.predict(X_test)
    y_pred_var = ((y_pred_upper - y_pred_lower) / 4) ** 2
else:
    residuals = y_test - y_pred
    y_pred_var = np.full_like(y_pred, fill_value=np.var(residuals))

# ベイズ最適化で最適な閾値を再度求める
min_pred = y_pred.min()
max_pred = y_pred.max()
space = [Real(min_pred, max_pred, name=f't{i}') for i in range(7)]

@use_named_args(space)
def objective(**thresholds):
    thresh = np.array([thresholds[f't{i}'] for i in range(7)])
    thresh.sort()
    y_pred_class = np.digitize(y_pred, bins=thresh) + 1
    kappa = cohen_kappa_score(y_test, y_pred_class, weights='quadratic')
    return -kappa

res = gp_minimize(objective, space, n_calls=50, random_state=42)
optimal_thresholds = np.array(res.x)
optimal_thresholds.sort()
y_pred_class = np.digitize(y_pred, bins=optimal_thresholds) + 1

# 各データポイントについて、予測クラスを取得
predicted_classes = y_pred_class

# 各データポイントについて、期待損失を計算
def calculate_expected_loss(y_pred, y_pred_var, thresholds, predicted_classes):
    expected_losses = []
    for i in range(len(y_pred)):
        mu = y_pred[i]
        sigma = np.sqrt(y_pred_var[i])
        # クラスの境界を定義
        class_boundaries = [float('-inf')] + list(thresholds) + [float('inf')]
        # 各クラスに属する確率を計算
        class_probs = []
        for j in range(len(class_boundaries)-1):
            prob = norm.cdf(class_boundaries[j+1], mu, sigma) - norm.cdf(class_boundaries[j], mu, sigma)
            class_probs.append(prob)
        class_probs = np.array(class_probs)
        # 予測クラスとの差の絶対値に比例する損失額を設定
        losses = np.array([abs(predicted_classes[i] - (j+1)) * 10000 for j in range(len(class_probs))])
        # 期待損失を計算
        expected_loss = np.sum(class_probs * losses)
        expected_losses.append(expected_loss)
    return np.array(expected_losses)

expected_losses = calculate_expected_loss(y_pred, y_pred_var, optimal_thresholds, predicted_classes)

# 人手判定のレコメンデーション
def recommend_human_assessment(expected_losses, human_cost=50000):
    human_recommended = expected_losses > human_cost
    return human_recommended

human_recommended = recommend_human_assessment(expected_losses)

print(f"\nNumber of cases recommended for human assessment: {np.sum(human_recommended)}")
print(f"Percentage of cases recommended for human assessment: {np.mean(human_recommended) * 100:.2f}%")

# 混同行列の表示
conf_matrix = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(1,9), yticklabels=range(1,9))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 他の評価指標の計算
accuracy = accuracy_score(y_test, predicted_classes)
precision = precision_score(y_test, predicted_classes, average='weighted')
recall = recall_score(y_test, predicted_classes, average='weighted')
f1 = f1_score(y_test, predicted_classes, average='weighted')
kappa = cohen_kappa_score(y_test, predicted_classes, weights='quadratic')

print(f"\nEvaluation Metrics for Best Model ({best_model_name}):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Quadratic Weighted Kappa: {kappa:.4f}")

# 結果の保存
results_df = pd.DataFrame({
    'True_Value': y_test,
    'Predicted_Value': predicted_classes,
    'Expected_Loss': expected_losses,
    'Human_Assessment_Recommended': human_recommended
})
results_df.to_csv('final_predictions_with_recommendations.csv', index=False)
print("\nFinal predictions with human assessment recommendations have been saved to 'final_predictions_with_recommendations.csv'")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, cohen_kappa_score,
                             confusion_matrix, f1_score, accuracy_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# 回帰モデルをインポート
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

# ベイズ最適化用のライブラリ
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# ブートストラップ用
from sklearn.utils import resample

# statsmodelsをインポート
import statsmodels.api as sm
from scipy.stats import norm

# データの読み込み
train = pd.read_csv('train.csv')

# 目的変数と説明変数の定義
X = train.drop(['Id', 'Response'], axis=1)
y = train['Response']

# カテゴリカルと数値データの分離
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 前処理の定義
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの定義
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(random_state=42),
    'Ridge': Ridge(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42)
}

# グリッドサーチのパラメータ
param_grids = {
    'Linear Regression': {},
    'Lasso': {'model__alpha': [0.001, 0.01, 0.1, 1]},
    'Ridge': {'model__alpha': [0.001, 0.01, 0.1, 1]},
    'Decision Tree': {'model__max_depth': [5, 10, 15, None]},
    'Random Forest': {'model__n_estimators': [100, 200], 'model__max_depth': [10, 20, None]},
    'LightGBM': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1]}
}

results = []

for name, model in models.items():
    print(f"Training {name}...")
    # パイプラインの作成
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
    
    # グリッドサーチの実行
    grid_search = GridSearchCV(clf, param_grids.get(name, {}), cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # テストデータでの予測
    y_pred = best_model.predict(X_test)
    
    # モデルごとの予測分散の推定
    if name == 'Linear Regression':
        # statsmodelsで予測の分散を取得
        X_train_sm = sm.add_constant(preprocessor.fit_transform(X_train))
        ols = sm.OLS(y_train, X_train_sm).fit()
        X_test_sm = sm.add_constant(preprocessor.transform(X_test))
        predictions = ols.get_prediction(X_test_sm)
        pred_summary = predictions.summary_frame(alpha=0.05)
        y_pred_var = pred_summary['mean_se'] ** 2  # 標準誤差の二乗が分散
        y_pred = pred_summary['mean']
    elif name == 'Lasso':
        # ブートストラップによる予測分散の推定
        n_bootstraps = 100
        y_preds_bootstrap = np.zeros((n_bootstraps, len(X_test)))
        for i in range(n_bootstraps):
            X_sample, y_sample = resample(X_train, y_train)
            clf_boot = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('model', Lasso(alpha=best_model.named_steps['model'].alpha, random_state=42))])
            clf_boot.fit(X_sample, y_sample)
            y_preds_bootstrap[i] = clf_boot.predict(X_test)
        y_pred = np.mean(y_preds_bootstrap, axis=0)
        y_pred_var = np.var(y_preds_bootstrap, axis=0)
    elif name == 'Random Forest':
        # 各決定木の予測の分散を取得
        model = best_model.named_steps['model']
        all_predictions = np.array([tree.predict(best_model.named_steps['preprocessor'].transform(X_test)) for tree in model.estimators_])
        y_pred = np.mean(all_predictions, axis=0)
        y_pred_var = np.var(all_predictions, axis=0)
    elif name == 'LightGBM':
        # クォンタイル回帰で予測区間を取得し、分散を近似
        # 下位予測（5%点）
        params_lower = {
            'objective': 'quantile',
            'alpha': 0.05,
            'verbose': -1
        }
        model_lower = LGBMRegressor(**params_lower, random_state=42)
        model_lower.fit(preprocessor.fit_transform(X_train), y_train)
        y_pred_lower = model_lower.predict(preprocessor.transform(X_test))
        
        # 上位予測（95%点）
        params_upper = {
            'objective': 'quantile',
            'alpha': 0.95,
            'verbose': -1
        }
        model_upper = LGBMRegressor(**params_upper, random_state=42)
        model_upper.fit(preprocessor.fit_transform(X_train), y_train)
        y_pred_upper = model_upper.predict(preprocessor.transform(X_test))
        
        # 中央予測
        y_pred = best_model.predict(X_test)
        y_pred_var = ((y_pred_upper - y_pred_lower) / 4) ** 2  # 標準偏差を近似し、二乗して分散に
    else:
        # その他のモデル（分散を平均二乗誤差で近似）
        residuals = y_test - y_pred
        y_pred_var = np.full_like(y_pred, fill_value=np.var(residuals))
    
    # 閾値の最適化（ベイズ最適化）
    from skopt.space import Space
    from skopt.space import Real
    from skopt.utils import use_named_args
    
    # 予測値の範囲を取得
    min_pred = y_pred.min()
    max_pred = y_pred.max()
    
    # 最適化するパラメータ空間を定義
    space = [Real(min_pred, max_pred, name=f't{i}') for i in range(7)]
    
    @use_named_args(space)
    def objective(**thresholds):
        # 閾値をソート
        thresh = np.array([thresholds[f't{i}'] for i in range(7)])
        thresh.sort()
        
        # 閾値に基づいてクラスに割り当て
        y_pred_class = np.digitize(y_pred, bins=thresh) + 1  # クラスは1から8
        
        # Quadratic Weighted Kappaの計算
        kappa = cohen_kappa_score(y_test, y_pred_class, weights='quadratic')
        return -kappa  # 最小化するので負にする
    
    # ベイズ最適化の実行
    res = gp_minimize(objective, space, n_calls=50, random_state=42)
    
    # 最適な閾値を取得
    optimal_thresholds = np.array(res.x)
    optimal_thresholds.sort()
    
    # 最適な閾値でクラスに割り当て
    y_pred_class = np.digitize(y_pred, bins=optimal_thresholds) + 1  # クラスは1から8
    
    # 評価指標の計算
    mae = mean_absolute_error(y_test, y_pred_class)
    mse = mean_squared_error(y_test, y_pred_class)
    kappa = cohen_kappa_score(y_test, y_pred_class, weights='quadratic')
    accuracy = accuracy_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class, average='weighted')
    precision = precision_score(y_test, y_pred_class, average='weighted')
    recall = recall_score(y_test, y_pred_class, average='weighted')
    
    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Quadratic Weighted Kappa': kappa
    })

# 結果の表示
results_df = pd.DataFrame(results)
print("\nModel Comparison Results:")
print(results_df.to_string(index=False))
results_df.to_csv('model_comparison_results.csv', index=False)

# 最良のモデルを選択（Quadratic Weighted Kappaが最大のモデル）
best_model_name = results_df.loc[results_df['Quadratic Weighted Kappa'].idxmax(), 'Model']
print(f"\nBest Model: {best_model_name}")

# 最良モデルでの詳細な評価
best_model = models[best_model_name]
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', best_model)])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# モデルごとの予測分散の推定（最良モデル）
if best_model_name == 'Linear Regression':
    X_train_sm = sm.add_constant(preprocessor.fit_transform(X_train))
    ols = sm.OLS(y_train, X_train_sm).fit()
    X_test_sm = sm.add_constant(preprocessor.transform(X_test))
    predictions = ols.get_prediction(X_test_sm)
    pred_summary = predictions.summary_frame(alpha=0.05)
    y_pred_var = pred_summary['mean_se'] ** 2
    y_pred = pred_summary['mean']
elif best_model_name == 'Lasso':
    n_bootstraps = 100
    y_preds_bootstrap = np.zeros((n_bootstraps, len(X_test)))
    for i in range(n_bootstraps):
        X_sample, y_sample = resample(X_train, y_train)
        clf_boot = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', Lasso(alpha=clf.named_steps['model'].alpha, random_state=42))])
        clf_boot.fit(X_sample, y_sample)
        y_preds_bootstrap[i] = clf_boot.predict(X_test)
    y_pred = np.mean(y_preds_bootstrap, axis=0)
    y_pred_var = np.var(y_preds_bootstrap, axis=0)
elif best_model_name == 'Random Forest':
    model = clf.named_steps['model']
    all_predictions = np.array([tree.predict(clf.named_steps['preprocessor'].transform(X_test)) for tree in model.estimators_])
    y_pred = np.mean(all_predictions, axis=0)
    y_pred_var = np.var(all_predictions, axis=0)
elif best_model_name == 'LightGBM':
    # クォンタイル回帰で予測区間を取得し、分散を近似
    params_lower = {
        'objective': 'quantile',
        'alpha': 0.05,
        'verbose': -1
    }
    model_lower = LGBMRegressor(**params_lower, random_state=42)
    model_lower.fit(preprocessor.fit_transform(X_train), y_train)
    y_pred_lower = model_lower.predict(preprocessor.transform(X_test))
    
    params_upper = {
        'objective': 'quantile',
        'alpha': 0.95,
        'verbose': -1
    }
    model_upper = LGBMRegressor(**params_upper, random_state=42)
    model_upper.fit(preprocessor.fit_transform(X_train), y_train)
    y_pred_upper = model_upper.predict(preprocessor.transform(X_test))
    
    y_pred = clf.predict(X_test)
    y_pred_var = ((y_pred_upper - y_pred_lower) / 4) ** 2
else:
    residuals = y_test - y_pred
    y_pred_var = np.full_like(y_pred, fill_value=np.var(residuals))

# ベイズ最適化で最適な閾値を再度求める
min_pred = y_pred.min()
max_pred = y_pred.max()
space = [Real(min_pred, max_pred, name=f't{i}') for i in range(7)]

@use_named_args(space)
def objective(**thresholds):
    thresh = np.array([thresholds[f't{i}'] for i in range(7)])
    thresh.sort()
    y_pred_class = np.digitize(y_pred, bins=thresh) + 1
    kappa = cohen_kappa_score(y_test, y_pred_class, weights='quadratic')
    return -kappa

res = gp_minimize(objective, space, n_calls=50, random_state=42)
optimal_thresholds = np.array(res.x)
optimal_thresholds.sort()
y_pred_class = np.digitize(y_pred, bins=optimal_thresholds) + 1

# 各データポイントについて、予測クラスを取得
predicted_classes = y_pred_class

# 各データポイントについて、期待損失を計算
def calculate_expected_loss(y_pred, y_pred_var, thresholds, predicted_classes):
    expected_losses = []
    for i in range(len(y_pred)):
        mu = y_pred[i]
        sigma = np.sqrt(y_pred_var[i])
        # クラスの境界を定義
        class_boundaries = [float('-inf')] + list(thresholds) + [float('inf')]
        # 各クラスに属する確率を計算
        class_probs = []
        for j in range(len(class_boundaries)-1):
            prob = norm.cdf(class_boundaries[j+1], mu, sigma) - norm.cdf(class_boundaries[j], mu, sigma)
            class_probs.append(prob)
        class_probs = np.array(class_probs)
        # 予測クラスとの差の絶対値に比例する損失額を設定
        losses = np.array([abs(predicted_classes[i] - (j+1)) * 10000 for j in range(len(class_probs))])
        # 期待損失を計算
        expected_loss = np.sum(class_probs * losses)
        expected_losses.append(expected_loss)
    return np.array(expected_losses)

expected_losses = calculate_expected_loss(y_pred, y_pred_var, optimal_thresholds, predicted_classes)

# 人手判定のレコメンデーション
def recommend_human_assessment(expected_losses, human_cost=50000):
    human_recommended = expected_losses > human_cost
    return human_recommended

human_recommended = recommend_human_assessment(expected_losses)

print(f"\nNumber of cases recommended for human assessment: {np.sum(human_recommended)}")
print(f"Percentage of cases recommended for human assessment: {np.mean(human_recommended) * 100:.2f}%")

# 混同行列の表示
conf_matrix = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(1,9), yticklabels=range(1,9))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 他の評価指標の計算
accuracy = accuracy_score(y_test, predicted_classes)
precision = precision_score(y_test, predicted_classes, average='weighted')
recall = recall_score(y_test, predicted_classes, average='weighted')
f1 = f1_score(y_test, predicted_classes, average='weighted')
kappa = cohen_kappa_score(y_test, predicted_classes, weights='quadratic')

print(f"\nEvaluation Metrics for Best Model ({best_model_name}):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Quadratic Weighted Kappa: {kappa:.4f}")

# 結果の保存
results_df = pd.DataFrame({
    'True_Value': y_test,
    'Predicted_Value': predicted_classes,
    'Expected_Loss': expected_losses,
    'Human_Assessment_Recommended': human_recommended
})
results_df.to_csv('final_predictions_with_recommendations.csv', index=False)
print("\nFinal predictions with human assessment recommendations have been saved to 'final_predictions_with_recommendations.csv'")
