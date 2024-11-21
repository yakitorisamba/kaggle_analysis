def prepare_category_dict(category_dict):
    """
    分類辞書の検索を高速化するための前処理
    """
    # 辞書をキーワードでインデックス化
    return {row.iloc[0]: row.iloc[1] for _, row in category_dict.iterrows()}

def classify_content_batch(texts, category_dict_map):
    """
    テキストを一括で高速分類する関数
    
    Args:
        texts (Series): 分類対象のテキスト群
        category_dict_map (dict): インデックス化された分類辞書
        
    Returns:
        tuple: (小分類リスト, 中分類リスト)
    """
    small_categories = []
    medium_categories = []
    keywords = list(category_dict_map.keys())
    
    for text in texts:
        if pd.isna(text):
            small_categories.append('その他')
            medium_categories.append('その他')
            continue
            
        text = str(text)
        found = False
        for keyword in keywords:
            if keyword in text:
                small_categories.append(keyword)
                medium_categories.append(category_dict_map[keyword])
                found = True
                break
        
        if not found:
            small_categories.append('その他')
            medium_categories.append('その他')
            
    return small_categories, medium_categories

def main():
    print("Loading category dictionary...")
    category_dict = load_category_dictionary('category_dict.csv')
    category_dict_map = prepare_category_dict(category_dict)
    
    print("Reading main data files...")
    main_df = read_csv_in_chunks('temp/*.csv')
    
    print("Classifying contents...")
    small_cats, medium_cats = classify_content_batch(main_df['content1'], category_dict_map)
    main_df['small_category'] = small_cats
    main_df['medium_category'] = medium_cats
    
    # content2の苦情チェック
    main_df['is_complaint'] = main_df['content2'].apply(check_complaint)
    
    # 以下は同じ...
