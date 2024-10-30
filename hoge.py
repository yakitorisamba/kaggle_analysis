# data_processing_utils.py

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

def load_data(file_paths: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """前のコードと同じ"""
    # 省略（変更なし）

def create_prediction_table(individual_df: pd.DataFrame, corporate_df: pd.DataFrame) -> pd.DataFrame:
    """前のコードと同じ"""
    # 省略（変更なし）

def melt_features_and_shap(df: pd.DataFrame, id_vars: List[str]) -> pd.DataFrame:
    """
    特徴量とSHAP値のmelt処理を一括で行う関数
    
    Args:
        df (pd.DataFrame): 入力データフレーム
        id_vars (List[str]): ID変数のリスト
    
    Returns:
        pd.DataFrame: 特徴量とSHAP値がマージされたデータフレーム
    """
    # 特徴量とSHAP値の列を特定
    feature_cols = df.columns.difference(id_vars + ['hugo1', 'hugohugo', 'hogehuga'])
    
    # 一度のmeltで両方の値を取得
    melted = pd.melt(df[id_vars + list(feature_cols)],
                     id_vars=id_vars,
                     var_name='特徴量名称')
    
    # 特徴量名称でグループ化して、SHAP値と数値を区別
    merged = melted.copy()
    merged['特徴量_数値'] = merged['value']
    merged['特徴量_shap値'] = merged['value']
    merged = merged.drop('value', axis=1)
    
    return merged

def calculate_shap_statistics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    SHAP値の統計量を計算する関数
    
    Args:
        df (pd.DataFrame): マージされたデータフレーム
    
    Returns:
        Dict[str, pd.DataFrame]: 各種統計量を含むデータフレーム
    """
    # SHAP値の平均を計算（より効率的な方法）
    shap_stats = df.groupby(['特徴量名称', '特徴量_数値'])['特徴量_shap値'].agg({
        '特徴量_shap値平均': 'mean',
        '特徴量_shap値標準偏差': 'std',
        'データ数': 'count'
    }).reset_index()
    
    # 絶対値の計算と追加の統計量
    shap_stats['|特徴量_shap値平均|'] = shap_stats['特徴量_shap値平均'].abs()
    
    # 特徴量ごとの閾値を計算（より詳細な統計情報を含む）
    thresholds = shap_stats.groupby('特徴量名称').agg({
        '|特徴量_shap値平均|': ['min', 'max', 'mean'],
        '特徴量_shap値標準偏差': 'mean',
        'データ数': 'sum'
    }).reset_index()
    
    # カラム名を整理
    thresholds.columns = ['特徴量名称', 'shap値閾値', 'shap値最大', 'shap値平均', 'shap値標準偏差', '総データ数']
    
    return {
        'detailed_stats': shap_stats,
        'thresholds': thresholds
    }

def create_prediction_basis_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    予測根拠テーブルを作成する関数（改善版）
    """
    # ID変数の定義
    id_vars = ['期待値', '基準年月', 'コード', 'Mコード']
    
    # 改善されたmelt処理を使用
    merged_df = melt_features_and_shap(df, id_vars)
    
    # SHAP値の絶対値を計算
    merged_df['特徴量_shap値絶対値'] = merged_df['特徴量_shap値'].abs()
    
    # より効率的なランク付け
    merged_df['rank'] = merged_df.groupby(['基準年月', 'コード', 'Mコード'])['特徴量_shap値絶対値'].rank(
        method='min',
        ascending=False
    )
    
    return merged_df

def calculate_feature_correlations(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    特徴量の相関係数を計算する関数（改善版）
    """
    # ID変数の定義
    id_vars = ['期待値', '基準年月', 'コード', 'Mコード']
    
    # 改善されたmelt処理を使用
    merged_df = melt_features_and_shap(df, id_vars)
    
    # 相関係数の計算（より効率的な方法）
    correlations = merged_df.groupby('特徴量名称').apply(
        lambda x: pd.Series({
            '相関係数': x['特徴量_数値'].corr(x['特徴量_shap値']),
            'データ数': len(x),
            'p値': stats.pearsonr(x['特徴量_数値'], x['特徴量_shap値'])[1]
        })
    ).reset_index()
    
    return correlations, merged_df
