import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# 分類モデルをインポート
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# データの読み込みと前処理
train = pd.read_csv('train.csv')

# 目的変数と説明変数の定義
X = train.drop(['Id', 'Response'], axis=1)
y = train['Response'] - 1  # クラスラベルを0から始まるように変更

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
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVC': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'MLP': MLPClassifier(random_state=42, max_iter=1000),
    'LightGBM': LGBMClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0)
}

# グリッドサーチのパラメータ
param_grids = {
    'Logistic Regression': {'model__C': [0.01, 0.1, 1, 10]},
    'Decision Tree': {'model__max_depth': [5, 10, 15, None]},
    'Random Forest': {'model__n_estimators': [100, 200], 'model__max_depth': [10, 20, None]},
    'Gradient Boosting': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1]},
    'SVC': {'model__C': [0.1, 1, 10], 'model__kernel': ['rbf', 'linear']},
    'KNN': {'model__n_neighbors': [3, 5, 7], 'model__weights': ['uniform', 'distance']},
    'MLP': {'model__hidden_layer_sizes': [(50,), (100,), (50, 50)], 'model__alpha': [0.0001, 0.001]},
    'LightGBM': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1], 'model__num_leaves': [31, 127]},
    'XGBoost': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1], 'model__max_depth': [3, 6]},
    'CatBoost': {'model__iterations': [100, 200], 'model__learning_rate': [0.01, 0.1], 'model__depth': [4, 6]}
}

results = []

for name, model in models.items():
    print(f"Training {name}...")
    # パイプラインの作成
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
    
    # グリッドサーチの実行
    grid_search = GridSearchCV(clf, param_grids[name], cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    # 評価指標の計算
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Quadratic Weighted Kappa': kappa
    })

results_df = pd.DataFrame(results)
print("\nModel Comparison Results:")
print(results_df.to_string(index=False))
results_df.to_csv('model_comparison_results.csv', index=False)

# 最良のモデルを選択
best_model_name = results_df.loc[results_df['Quadratic Weighted Kappa'].idxmax(), 'Model']
print(f"\nBest Model: {best_model_name}")

# 最良モデルの再学習
best_model = models[best_model_name]
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', best_model.set_params(**grid_search.best_params_['model']))])
clf.fit(X_train, y_train)

# テストデータに対する予測
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

# 不確実性の計算（エントロピー）
def calculate_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)

uncertainty = calculate_entropy(y_pred_proba)

# 誤分類の損失モデル
def calculate_loss(y_true, y_pred):
    return np.abs(y_true - y_pred)

loss = calculate_loss(y_test, y_pred)

# 人手判定のレコメンデーション機能
def recommend_human_assessment(uncertainty, loss, human_cost=50000):
    # 不確実性と損失を組み合わせて人手判定を推奨
    expected_loss = uncertainty * loss
    human_recommended = expected_loss > human_cost
    return human_recommended

# 人手判定の推奨
human_recommended = recommend_human_assessment(uncertainty, loss)

print(f"\nNumber of cases recommended for human assessment: {np.sum(human_recommended)}")
print(f"Percentage of cases recommended for human assessment: {np.mean(human_recommended) * 100:.2f}%")

# 結果の保存
results_df = pd.DataFrame({
    'True_Value': y_test + 1,  # クラスラベルを元に戻す
    'Predicted_Value': y_pred + 1,
    'Uncertainty': uncertainty,
    'Loss': loss,
    'Human_Assessment_Recommended': human_recommended
})
results_df.to_csv('final_predictions_with_recommendations.csv', index=False)
print("\nFinal predictions with human assessment recommendations have been saved to 'final_predictions_with_recommendations.csv'")
