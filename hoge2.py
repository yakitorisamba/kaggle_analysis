# main.py

from data_processing_utils import *
import pandas as pd
from scipy import stats

def main():
    # ファイルパスの設定
    file_paths = {
        'individual': 'predict_individual.csv',
        'corporate': 'predict_corporate.csv',
        'trained': 'trained_corporate.csv'
    }
    
    # 1. データの読み込み
    predict_individual, predict_corporate, trained_corporate = load_data(file_paths)
    
    # 2. 予測結果テーブルの作成と出力
    prediction_table = create_prediction_table(predict_individual, predict_corporate)
    prediction_table.to_csv('予測結果テーブル.csv', index=False, encoding='cp932')
    
    # 3. 予測根拠テーブルの作成と出力
    prediction_basis = create_prediction_basis_table(predict_corporate)
    prediction_basis.to_csv('予測根拠テーブル.csv', index=False, encoding='cp932')
    
    # 4. 特徴量傾向テーブルの作成
    correlations, merged_data = calculate_feature_correlations(trained_corporate)
    stats_dict = calculate_shap_statistics(merged_data)
    
    # 相関係数とSHAP値統計量のマージ
    feature_trends = pd.merge(
        correlations,
        stats_dict['thresholds'],
        on='特徴量名称'
    )
    
    # 5. 特徴量傾向テーブルの出力（より詳細な情報を含む）
    with pd.ExcelWriter('特徴量傾向テーブル.xlsx') as writer:
        feature_trends.to_excel(writer, sheet_name='概要', index=False)
        stats_dict['detailed_stats'].to_excel(writer, sheet_name='詳細統計', index=False)

if __name__ == "__main__":
    main()
