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

def analyze_large_csv(filepath):
    # データの読み込みと前処理
    df = read_large_csv(filepath)
    
    # 2020年のデータを削除
    df = df[df['event_datetime'].dt.year != 2020]
    
    # 家族登録と契約成立の関係分析
    def analyze_family_contract():
        family_stats = df[df['small_category'].str.contains('家族登録', na=False)].compute()
        
        contract_by_family = pd.crosstab(
            family_stats['small_category'],
            family_stats['medium_category'].str.contains('契約成立', na=False)
        )
        
        fig = px.bar(contract_by_family, 
                    title='家族登録有無による契約成立の比較',
                    labels={'value': '件数', 'small_category': '家族登録', 'medium_category': '契約成立'})
        return fig
    
    # 解約の有無によるコンタクト数の分析
    def analyze_contacts_by_cancellation():
        contact_stats = df.groupby('customer_id').agg({
            'medium_category': lambda x: x.str.contains('解約', na=False).any(),
            'customer_id': 'count'
        }).compute()
        
        fig = px.box(contact_stats, 
                    x='medium_category',
                    y='customer_id',
                    title='解約有無別のコンタクト数分布',
                    labels={'medium_category': '解約有無', 'customer_id': 'コンタクト数'})
        return fig
    
    # POL_NOとELEC_FLGの関係分析
    def analyze_pol_elec():
        pol_stats = df[df['POL_NO'].notna()].compute()
        
        cross_tab = pd.crosstab(
            [pol_stats['ELEC_FLG'], 
             pol_stats['medium_category'].str.contains('契約成立', na=False)],
            pol_stats['medium_category'].str.contains('解約', na=False)
        )
        
        fig = px.bar(cross_tab.reset_index(), 
                    title='ELEC_FLGカテゴリ別の契約成立・解約状況',
                    barmode='group')
        return fig
    
    # クレーム（is_complaint）と契約・解約の関係分析
    def analyze_complaints():
        complaint_stats = df.groupby('is_complaint').agg({
            'medium_category': lambda x: (x.str.contains('契約成立', na=False).sum(),
                                       x.str.contains('解約', na=False).sum())
        }).compute()
        
        fig = px.bar(complaint_stats, 
                    title='クレーム有無による契約成立・解約件数',
                    barmode='group')
        return fig
    
    # 営業担当者別のクレームと契約成立の関係
    def analyze_sales_performance():
        sales_stats = df.groupby(['sales_person_id', 'is_complaint']).agg({
            'medium_category': lambda x: x.str.contains('契約成立', na=False).sum()
        }).compute()
        
        fig = px.bar(sales_stats.reset_index(), 
                    x='sales_person_id',
                    y='medium_category',
                    color='is_complaint',
                    title='営業担当者別クレーム有無と契約成立数',
                    barmode='group')
        return fig
    
    # 満期関連の分析
    def analyze_maturity():
        maturity_stats = df.groupby(['customer_id']).agg({
            'medium_category': lambda x: x.str.contains('満期', na=False).any(),
            'is_complaint': 'sum',
            'customer_id': 'count'
        }).compute()
        
        fig = px.scatter(maturity_stats, 
                        x='is_complaint',
                        y='customer_id',
                        color='medium_category',
                        title='満期有無とクレーム数、コンタクト数の関係',
                        labels={'is_complaint': 'クレーム数',
                               'customer_id': 'コンタクト数',
                               'medium_category': '満期有無'})
        return fig
    
    return {
        'family_contract': analyze_family_contract(),
        'contacts_cancellation': analyze_contacts_by_cancellation(),
        'pol_elec': analyze_pol_elec(),
        'complaints': analyze_complaints(),
        'sales_performance': analyze_sales_performance(),
        'maturity': analyze_maturity()
    }

# 使用例
# filepath = 'your_large_csv_file.csv'
# results = analyze_large_csv(filepath)
# 各グラフの表示
# for name, fig in results.items():
#     fig.show()
