import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

def read_large_csv(filepath):
    """メモリ効率の良いCSVファイルの読み込み"""
    # スキャンオプションの設定
    scan_options = pl.ScanCsvOptions(
        chunk_size=10000,  # チャンクサイズ
        low_memory=True
    )
    
    # LazyFrameとして読み込み
    df = pl.scan_csv(
        filepath,
        low_memory=True
    )
    
    # event_datetimeを処理し、2020年のデータを除外
    df = df.with_columns([
        pl.col('event_datetime').str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
    ]).filter(
        pl.col('event_datetime').dt.year() != 2020
    )
    
    return df

def analyze_family_contract(df):
    """家族登録と契約成立の関係を分析"""
    # LazyFrameでの処理
    customer_stats = df.with_columns([
        pl.col('small_category').str.contains('家族登録').alias('has_family'),
        pl.col('medium_category').str.contains('契約成立').alias('has_contract')
    ]).groupby('customer_id').agg([
        pl.col('has_family').max(),
        pl.col('has_contract').sum()
    ]).collect()  # ここでのみ実際のデータを読み込み
    
    # 可視化と統計処理は以前と同じ
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=customer_stats.to_pandas(), x='has_family', y='has_contract')
    plt.title('家族登録有無による契約成立数の分布')
    plt.xlabel('家族登録')
    plt.ylabel('契約成立数')
    
    stats = customer_stats.group_by('has_family').agg([
        pl.col('has_contract').count().alias('count'),
        pl.col('has_contract').mean().alias('mean'),
        pl.col('has_contract').median().alias('median'),
        pl.col('has_contract').std().alias('std')
    ])
    
    return plt.gcf(), stats

def analyze_customer_metrics(df):
    """解約、満期とクレーム、コンタクト数の関係を分析"""
    # LazyFrameでの処理
    customer_stats = df.with_columns([
        pl.col('medium_category').str.contains('解約').alias('has_cancellation'),
        pl.col('medium_category').str.contains('満期').alias('has_maturity')
    ]).groupby('customer_id').agg([
        pl.col('has_cancellation').max(),
        pl.col('has_maturity').max(),
        pl.col('is_complaint').sum().alias('complaint_count'),
        pl.count().alias('contact_count')
    ]).collect()
    
    # 可視化は以前と同じ
    fig_cancel, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    customer_stats_pd = customer_stats.to_pandas()
    
    sns.boxplot(data=customer_stats_pd, x='has_cancellation', y='complaint_count', ax=ax1)
    ax1.set_title('解約有無によるクレーム数の分布')
    ax1.set_xlabel('解約有無')
    ax1.set_ylabel('クレーム数')
    
    sns.boxplot(data=customer_stats_pd, x='has_cancellation', y='contact_count', ax=ax2)
    ax2.set_title('解約有無によるコンタクト数の分布')
    ax2.set_xlabel('解約有無')
    ax2.set_ylabel('コンタクト数')
    
    return {
        'cancellation_plot': fig_cancel,
        'stats': customer_stats
    }

def analyze_polno_categories(df):
    """POL_NOの値ごとの分析"""
    # LazyFrameでの処理
    customer_stats = df.filter(pl.col('POL_NO').is_not_null()).with_columns([
        pl.col('medium_category').str.contains('契約成立').alias('has_contract'),
        pl.col('medium_category').str.contains('解約').alias('has_cancellation')
    ]).groupby(['customer_id', 'POL_NO']).agg([
        pl.col('has_contract').sum().alias('contract_count'),
        pl.col('has_cancellation').sum().alias('cancellation_count'),
        pl.col('is_complaint').sum().alias('complaint_count'),
        pl.count().alias('contact_count')
    ]).collect()
    
    # 可視化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    customer_stats_pd = customer_stats.to_pandas()
    
    sns.boxplot(data=customer_stats_pd, x='POL_NO', y='contract_count', ax=ax1)
    ax1.set_title('POL_NO値ごとの契約成立数')
    
    sns.boxplot(data=customer_stats_pd, x='POL_NO', y='cancellation_count', ax=ax2)
    ax2.set_title('POL_NO値ごとの解約数')
    
    sns.boxplot(data=customer_stats_pd, x='POL_NO', y='complaint_count', ax=ax3)
    ax3.set_title('POL_NO値ごとのクレーム数')
    
    sns.boxplot(data=customer_stats_pd, x='POL_NO', y='contact_count', ax=ax4)
    ax4.set_title('POL_NO値ごとのコンタクト数')
    
    return fig, customer_stats

def run_analysis(filepath):
    """分析の実行"""
    print("データを読み込んでいます...")
    df = read_large_csv(filepath)
    
    # 各分析を実行
    print("\n家族登録分析を実行中...")
    family_plot, family_stats = analyze_family_contract(df)
    plt.figure(family_plot.number)
    plt.savefig('family_analysis.png')
    plt.close()
    print(family_stats)
    
    print("\n顧客メトリクス分析を実行中...")
    metrics_results = analyze_customer_metrics(df)
    plt.figure(metrics_results['cancellation_plot'].number)
    plt.savefig('cancellation_analysis.png')
    plt.close()
    print(metrics_results['stats'])
    
    print("\nPOL_NO分析を実行中...")
    polno_plot, polno_stats = analyze_polno_categories(df)
    plt.figure(polno_plot.number)
    plt.savefig('polno_analysis.png')
    plt.close()
    print(polno_stats)

if __name__ == "__main__":
    filepath = "your_data.csv"
    run_analysis(filepath)
