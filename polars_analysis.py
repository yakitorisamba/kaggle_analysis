#%%
# polars version
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

def read_large_csv_polars(filepath):
    return pl.read_csv(
        filepath,
        try_parse_dates=True
    )

def analyze_family_contract_polars(df):
    # 家族登録と契約成立フラグを作成
    customer_stats = df.with_columns([
        pl.col('small_category').str.contains('家族登録').alias('has_family'),
        pl.col('medium_category').str.contains('契約成立').alias('has_contract')
    ]).groupby('customer_id').agg([
        pl.col('has_family').max(),
        pl.col('has_contract').sum()
    ]).collect()
    
    # 可視化
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=customer_stats.to_pandas(), x='has_family', y='has_contract')
    plt.title('家族登録有無による契約成立数の分布')
    plt.xlabel('家族登録')
    plt.ylabel('契約成立数')
    
    # 統計量
    stats = customer_stats.group_by('has_family').agg([
        pl.col('has_contract').mean(),
        pl.col('has_contract').median(),
        pl.col('has_contract').std()
    ])
    
    return plt.gcf(), stats

def analyze_customer_metrics_polars(df):
    # フラグ作成と集計
    customer_stats = df.with_columns([
        pl.col('medium_category').str.contains('解約').alias('has_cancellation'),
        pl.col('medium_category').str.contains('満期').alias('has_maturity')
    ]).groupby('customer_id').agg([
        pl.col('has_cancellation').max(),
        pl.col('has_maturity').max(),
        pl.col('is_complaint').sum(),
        pl.count().alias('contact_count')
    ]).collect()
    
    # 解約分析の可視化
    fig_cancel, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    customer_stats_pd = customer_stats.to_pandas()
    
    sns.boxplot(data=customer_stats_pd, x='has_cancellation', y='is_complaint', ax=ax1)
    ax1.set_title('解約有無によるクレーム数の分布')
    
    sns.boxplot(data=customer_stats_pd, x='has_cancellation', y='contact_count', ax=ax2)
    ax2.set_title('解約有無によるコンタクト数の分布')
    
    # 満期分析の可視化
    fig_maturity, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 12))
    
    sns.boxplot(data=customer_stats_pd, x='has_maturity', y='is_complaint', ax=ax3)
    ax3.set_title('満期有無によるクレーム数の分布')
    
    sns.boxplot(data=customer_stats_pd, x='has_maturity', y='contact_count', ax=ax4)
    ax4.set_title('満期有無によるコンタクト数の分布')
    
    return {
        'cancellation': fig_cancel,
        'maturity': fig_maturity,
        'stats': customer_stats.describe()
    }

def analyze_polno_polars(df):
    # フラグ作成と集計
    customer_stats = df.filter(pl.col('POL_NO').is_not_null()).with_columns([
        pl.col('medium_category').str.contains('契約成立').alias('has_contract'),
        pl.col('medium_category').str.contains('解約').alias('has_cancellation')
    ]).groupby(['customer_id', 'POL_NO']).agg([
        pl.col('has_contract').sum(),
        pl.col('has_cancellation').sum(),
        pl.col('is_complaint').sum(),
        pl.count().alias('contact_count')
    ]).collect()
    
    # 可視化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
    customer_stats_pd = customer_stats.to_pandas()
    
    sns.boxplot(data=customer_stats_pd, x='POL_NO', y='has_contract', ax=ax1)
    ax1.set_title('POL_NO値ごとの契約成立数')
    
    sns.boxplot(data=customer_stats_pd, x='POL_NO', y='has_cancellation', ax=ax2)
    ax2.set_title('POL_NO値ごとの解約数')
    
    sns.boxplot(data=customer_stats_pd, x='POL_NO', y='is_complaint', ax=ax3)
    ax3.set_title('POL_NO値ごとのクレーム数')
    
    sns.boxplot(data=customer_stats_pd, x='POL_NO', y='contact_count', ax=ax4)
    ax4.set_title('POL_NO値ごとのコンタクト数')
    
    return fig, customer_stats.group_by('POL_NO').agg([
        pl.col('has_contract').mean(),
        pl.col('has_contract').median(),
        pl.col('has_contract').std(),
        pl.col('has_cancellation').mean(),
        pl.col('has_cancellation').median(),
        pl.col('has_cancellation').std(),
        pl.col('is_complaint').mean(),
        pl.col('is_complaint').median(),
        pl.col('is_complaint').std(),
        pl.col('contact_count').mean(),
        pl.col('contact_count').median(),
        pl.col('contact_count').std()
    ])

