#%%
import dask.dataframe as dd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def read_large_csv(filepath):
    # すべての列の型をstrで初期化
    dtypes = {col: 'string' for col in pd.read_csv(filepath, nrows=0).columns}
    
    # is_complaintの型を指定
    dtypes['is_complaint'] = 'bool'
    
    # event_datetimeはparse_datesで処理するため、dtypesから除外
    if 'event_datetime' in dtypes:
        del dtypes['event_datetime']
    
    # データ読み込み
    df = dd.read_csv(
        filepath,
        dtype=dtypes,
        parse_dates=['event_datetime'],
        assume_missing=True
    )
    
    return df

def analyze_family_contract(df):
    # customer_idごとに家族登録の有無を判定
    customer_family = df.groupby('customer_id').agg({
        'small_category': lambda x: x.str.contains('家族登録', na=False).max(),  # 家族登録の有無
    }).compute()
    
    # customer_idごとの契約成立件数を集計
    customer_contracts = df.groupby('customer_id').agg({
        'medium_category': lambda x: x.str.contains('契約成立', na=False).sum()  # 契約成立の件数
    }).compute()
    
    # データの結合
    analysis_df = pd.merge(
        customer_family,
        customer_contracts,
        left_index=True,
        right_index=True
    )
    
    # カラム名の変更
    analysis_df.columns = ['has_family_registration', 'contract_count']
    
    # 可視化1: 箱ひげ図で分布を比較
    fig1 = px.box(
        analysis_df,
        x='has_family_registration',
        y='contract_count',
        title='家族登録有無による顧客ごとの契約成立数の分布',
        labels={
            'has_family_registration': '家族登録',
            'contract_count': '契約成立数',
        }
    )
    
    # 可視化2: バイオリンプロットで分布の詳細を表示
    fig2 = px.violin(
        analysis_df,
        x='has_family_registration',
        y='contract_count',
        box=True,
        title='家族登録有無による顧客ごとの契約成立数の分布（詳細）',
        labels={
            'has_family_registration': '家族登録',
            'contract_count': '契約成立数',
        }
    )
    
    # 基本統計量の計算
    stats = analysis_df.groupby('has_family_registration')['contract_count'].agg([
        'count',
        'mean',
        'median',
        'std'
    ]).round(2)
    
    return {
        'box_plot': fig1,
        'violin_plot': fig2,
        'statistics': stats
    }

def analyze_customer_metrics(df):
    # 解約とクレーム、コンタクト数の分析
    def analyze_cancellation_impact():
        # まず解約フラグを作成
        df_with_flags = df.assign(
            has_cancellation=df['medium_category'].str.contains('解約', na=False)
        )
        
        # customer_idごとの集計
        customer_stats = df_with_flags.groupby('customer_id').agg({
            'has_cancellation': 'max',  # 解約の有無
            'is_complaint': 'sum',      # クレーム数
            'customer_id': 'size'       # コンタクト数
        }).compute()
        
        # 可視化1: クレーム数の分布
        fig1 = px.box(
            customer_stats,
            x='has_cancellation',
            y='is_complaint',
            title='解約有無による顧客ごとのクレーム数分布',
            labels={
                'has_cancellation': '解約有無',
                'is_complaint': 'クレーム数'
            }
        )
        
        # 可視化2: コンタクト数の分布
        fig2 = px.box(
            customer_stats,
            x='has_cancellation',
            y='customer_id',
            title='解約有無による顧客ごとのコンタクト数分布',
            labels={
                'has_cancellation': '解約有無',
                'customer_id': 'コンタクト数'
            }
        )
        
        return {'complaint_dist': fig1, 'contact_dist': fig2}
    
    # POL_NOの有無による分析
    def analyze_polno_impact():
        # POL_NOの有無フラグを作成
        df_with_flags = df.assign(
            has_polno=df['POL_NO'].notna(),
            has_cancellation=df['medium_category'].str.contains('解約', na=False)
        )
        
        # customer_idごとの集計
        customer_stats = df_with_flags.groupby('customer_id').agg({
            'has_polno': 'max',         # POL_NOの有無
            'has_cancellation': 'max',   # 解約の有無
            'is_complaint': 'sum',       # クレーム数
            'customer_id': 'size'        # コンタクト数
        }).compute()
        
        # 可視化: POL_NOの有無による各指標の分布
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('クレーム数分布', 'コンタクト数分布')
        )
        
        # クレーム数の箱ひげ図
        fig.add_trace(
            go.Box(
                x=customer_stats['has_polno'],
                y=customer_stats['is_complaint'],
                name='クレーム数'
            ),
            row=1, col=1
        )
        
        # コンタクト数の箱ひげ図
        fig.add_trace(
            go.Box(
                x=customer_stats['has_polno'],
                y=customer_stats['customer_id'],
                name='コンタクト数'
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=800, title_text="POL_NO有無による各指標の分布")
        
        return fig
    
    # 満期の分析
    def analyze_maturity_impact():
        # 満期フラグを作成
        df_with_flags = df.assign(
            has_maturity=df['medium_category'].str.contains('満期', na=False),
            has_cancellation=df['medium_category'].str.contains('解約', na=False)
        )
        
        # customer_idごとの集計
        customer_stats = df_with_flags.groupby('customer_id').agg({
            'has_maturity': 'max',       # 満期の有無
            'has_cancellation': 'max',    # 解約の有無
            'is_complaint': 'sum',        # クレーム数
            'customer_id': 'size'         # コンタクト数
        }).compute()
        
        # 可視化1: 満期有無とクレーム数の関係
        fig1 = px.box(
            customer_stats,
            x='has_maturity',
            y='is_complaint',
            title='満期有無による顧客ごとのクレーム数分布',
            labels={
                'has_maturity': '満期有無',
                'is_complaint': 'クレーム数'
            }
        )
        
        # 可視化2: 満期有無とコンタクト数の関係
        fig2 = px.box(
            customer_stats,
            x='has_maturity',
            y='customer_id',
            title='満期有無による顧客ごとのコンタクト数分布',
            labels={
                'has_maturity': '満期有無',
                'customer_id': 'コンタクト数'
            }
        )
        
        return {'complaint_dist': fig1, 'contact_dist': fig2}

    results = {
        'cancellation_analysis': analyze_cancellation_impact(),
        'polno_analysis': analyze_polno_impact(),
        'maturity_analysis': analyze_maturity_impact()
    }
    
    return results

def analyze_sales_performance(df):
    # 営業担当者別のクレームと契約成立の関係を分析
    df_with_flags = df.assign(
        has_contract=df['medium_category'].str.contains('契約成立', na=False)
    )
    
    sales_stats = df_with_flags.groupby(['sales_person_id', 'is_complaint']).agg({
        'has_contract': 'sum'  # 契約成立数
    }).compute()
    
    fig = px.bar(
        sales_stats.reset_index(),
        x='sales_person_id',
        y='has_contract',
        color='is_complaint',
        title='営業担当者別クレーム有無と契約成立数',
        labels={
            'sales_person_id': '営業担当者ID',
            'has_contract': '契約成立数',
            'is_complaint': 'クレーム有無'
        },
        barmode='group'
    )
    
    return fig

def analyze_large_csv(filepath):
    # データの読み込みと前処理
    df = read_large_csv(filepath)
    df = df[df['event_datetime'].dt.year != 2020]
    
    # 各分析の実行
    results = {
        'family_contract': analyze_family_contract(df),
        'customer_metrics': analyze_customer_metrics(df),
        'sales_performance': analyze_sales_performance(df)
    }
    
    return results

# 使用例
# filepath = 'your_csv_file.csv'
# results = analyze_large_csv(filepath)

# 結果の表示例
"""
# 家族登録分析
results['family_contract']['box_plot'].show()
results['family_contract']['violin_plot'].show()
print(results['family_contract']['statistics'])

# 顧客メトリクス分析
results['customer_metrics']['cancellation_analysis']['complaint_dist'].show()
results['customer_metrics']['cancellation_analysis']['contact_dist'].show()
results['customer_metrics']['polno_analysis'].show()
results['customer_metrics']['maturity_analysis']['complaint_dist'].show()
results['customer_metrics']['maturity_analysis']['contact_dist'].show()

# 営業パフォーマンス分析
results['sales_performance'].show()
"""
