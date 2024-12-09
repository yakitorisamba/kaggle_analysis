import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from scipy import stats

def analyze_customer_salesperson_metrics():
    """
    顧客とセールスマン指標の関係を分析する関数
    """
    # データ読み込み
    df = pl.read_csv("notifications.csv")
    
    # セールスマンごとの指標を計算
    salesperson_metrics = df.groupby('salesperson_id').agg([
        pl.col('customer_id').n_unique().alias('total_customers'),
        pl.col('notification_type2').eq('苦情').sum().alias('total_complaints'),
        # 見込み客情報がある場合の数
        pl.col('prospect_info').is_not_null().sum().alias('prospect_count'),
        # クレーム率（顧客あたり）
        (pl.col('notification_type2').eq('苦情').sum() / pl.col('customer_id').n_unique()).alias('complaint_rate')
    ])

    # 顧客ごとのヒストグラム分布を計算
    customer_hist = df.groupby('customer_id').agg([
        pl.col('notification_type1').count().alias('contact_count'),
        pl.col('salesperson_id').n_unique().alias('unique_salespersons'),
        pl.col('notification_type2').eq('苦情').sum().alias('complaint_count')
    ])

    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('セールスマン指標と顧客分布の関係')

    # 1. 見込み客数と顧客接触回数の関係
    sns.scatterplot(
        data=salesperson_metrics.to_pandas(),
        x='prospect_count',
        y='total_customers',
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('見込み客数と総顧客数の関係')
    axes[0, 0].set_xlabel('見込み客数')
    axes[0, 0].set_ylabel('総顧客数')

    # 2. クレーム率と顧客数の関係
    sns.scatterplot(
        data=salesperson_metrics.to_pandas(),
        x='complaint_rate',
        y='total_customers',
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('クレーム率と総顧客数の関係')
    axes[0, 1].set_xlabel('クレーム率')
    axes[0, 1].set_ylabel('総顧客数')

    # 3. 担当セールスマン数の分布
    sns.histplot(
        data=customer_hist.to_pandas(),
        x='unique_salespersons',
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('顧客あたりの担当セールスマン数分布')
    axes[1, 0].set_xlabel('担当セールスマン数')
    axes[1, 0].set_ylabel('顧客数')

    # 4. 接触回数とクレーム数の関係
    sns.scatterplot(
        data=customer_hist.to_pandas(),
        x='contact_count',
        y='complaint_count',
        ax=axes[1, 1]
    )
    axes[1, 1].set_title('接触回数とクレーム数の関係')
    axes[1, 1].set_xlabel('接触回数')
    axes[1, 1].set_ylabel('クレーム数')

    plt.tight_layout()
    plt.show()

    return salesperson_metrics, customer_hist

def analyze_customer_relationship():
    """
    受取人との続柄と解約率の関係を分析する関数
    """
    df = pl.read_csv("notifications.csv")
    
    # 続柄ごとの解約率を計算
    relationship_metrics = df.groupby('beneficiary_relationship').agg([
        pl.col('customer_id').n_unique().alias('total_customers'),
        pl.col('notification_type2').eq('解約').sum().alias('cancellations'),
        pl.col('notification_type2').eq('契約成立').sum().alias('new_contracts')
    ]).with_columns([
        (pl.col('cancellations') / pl.col('total_customers') * 100).alias('cancellation_rate')
    ])

    # 続柄ごとの解約率の可視化
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=relationship_metrics.sort('cancellation_rate', descending=True).to_pandas(),
        x='beneficiary_relationship',
        y='cancellation_rate'
    )
    plt.title('受取人との続柄別の解約率')
    plt.xlabel('続柄')
    plt.ylabel('解約率 (%)')
    plt.xticks(rotation=45)
    plt.show()

    # 続柄と契約継続期間の分析
    contract_duration = df.filter(
        pl.col('notification_type2').is_in(['契約成立', '解約'])
    ).groupby(['customer_id', 'beneficiary_relationship']).agg([
        pl.col('date').min().alias('start_date'),
        pl.col('date').max().alias('end_date')
    ]).with_columns([
        (pl.col('end_date') - pl.col('start_date')).dt.days().alias('duration_days')
    ])

    # 続柄別の契約継続期間の箱ひげ図
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=contract_duration.to_pandas(),
        x='beneficiary_relationship',
        y='duration_days'
    )
    plt.title('受取人との続柄別の契約継続期間')
    plt.xlabel('続柄')
    plt.ylabel('継続日数')
    plt.xticks(rotation=45)
    plt.show()

    return relationship_metrics, contract_duration

def analyze_salesperson_assignment_hypothesis():
    """
    担当セールスマン数に関する仮説検証
    
    仮説:
    1. 担当セールスマン数が多い顧客ほど解約リスクが高い
    2. 適切な担当セールスマン数（2-3人）の場合、顧客満足度が高い
    """
    df = pl.read_csv("notifications.csv")
    
    # 顧客ごとの指標を計算
    customer_metrics = df.groupby('customer_id').agg([
        pl.col('salesperson_id').n_unique().alias('salesperson_count'),
        pl.col('notification_type2').eq('解約').sum().alias('is_cancelled'),
        pl.col('notification_type2').eq('苦情').sum().alias('complaint_count'),
        pl.col('notification_type1').count().alias('interaction_count')
    ])

    # 担当者数による分類
    customer_metrics = customer_metrics.with_columns([
        pl.when(pl.col('salesperson_count') == 1)
        .then(pl.lit('1人'))
        .when(pl.col('salesperson_count') <= 3)
        .then(pl.lit('2-3人'))
        .otherwise(pl.lit('4人以上'))
        .alias('salesperson_category')
    ])

    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # 1. 担当者数カテゴリーごとの解約率
    cancellation_by_category = customer_metrics.groupby('salesperson_category').agg([
        pl.col('is_cancelled').sum() / pl.col('is_cancelled').count() * 100
    ]).sort('salesperson_category')
    
    sns.barplot(
        data=cancellation_by_category.to_pandas(),
        x='salesperson_category',
        y='is_cancelled',
        ax=axes[0]
    )
    axes[0].set_title('担当セールスマン数カテゴリーごとの解約率')
    axes[0].set_xlabel('担当セールスマン数')
    axes[0].set_ylabel('解約率 (%)')

    # 2. 担当者数カテゴリーごとの苦情数分布
    sns.boxplot(
        data=customer_metrics.to_pandas(),
        x='salesperson_category',
        y='complaint_count',
        ax=axes[1]
    )
    axes[1].set_title('担当セールスマン数カテゴリーごとの苦情数分布')
    axes[1].set_xlabel('担当セールスマン数')
    axes[1].set_ylabel('苦情数')

    plt.tight_layout()
    plt.show()

    # 統計的検定
    categories = customer_metrics.get_column('salesperson_category').unique().to_list()
    complaint_counts = [
        customer_metrics
        .filter(pl.col('salesperson_category') == cat)
        .get_column('complaint_count')
        .to_list()
        for cat in categories
    ]
    
    # Kruskal-Wallis検定
    h_stat, p_value = stats.kruskal(*complaint_counts)
    print(f"Kruskal-Wallis検定結果: H統計量 = {h_stat:.2f}, p値 = {p_value:.4f}")

    return customer_metrics

# 全ての分析を実行
def run_all_analyses():
    salesperson_metrics, customer_hist = analyze_customer_salesperson_metrics()
    relationship_metrics, contract_duration = analyze_customer_relationship()
    customer_metrics = analyze_salesperson_assignment_hypothesis()
    
    return {
        'salesperson_metrics': salesperson_metrics,
        'customer_hist': customer_hist,
        'relationship_metrics': relationship_metrics,
        'contract_duration': contract_duration,
        'customer_metrics': customer_metrics
    }
