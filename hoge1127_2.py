import csv
import os

def merge_consecutive_rows(csv_input_path, csv_output_path, key_columns):
    """
    CSVファイル内の連続する行を、複数の指定されたキー列に基づいてマージします。
    
    Parameters:
    csv_input_path (str): 入力CSVファイルのパス
    csv_output_path (str): 出力CSVファイルのパス
    key_columns (list): マージのキーとなる列名のリスト
    """
    # キー列が文字列として渡された場合、リストに変換
    if isinstance(key_columns, str):
        key_columns = [key_columns]
        
    temp_output_path = csv_output_path + '.tmp'
    current_fieldnames = set()
    
    # 初期フィールド名の取得
    with open(csv_input_path, 'r', newline='', encoding='cp932') as csvfile_in:
        reader = csv.DictReader(csvfile_in, quotechar='"')
        # キー列の存在確認
        missing_columns = [col for col in key_columns if col not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"指定されたキー列が見つかりません: {', '.join(missing_columns)}")
        original_fieldnames = reader.fieldnames.copy()
        current_fieldnames.update(original_fieldnames)

    with open(csv_input_path, 'r', newline='', encoding='cp932') as csvfile_in, \
         open(temp_output_path, 'w', newline='', encoding='cp932') as temp_output:
        
        reader = csv.DictReader(csvfile_in, quotechar='"')
        current_group = []
        current_key_values = None

        # 一時ファイルの初期化
        temp_fieldnames = list(original_fieldnames)
        temp_writer = csv.DictWriter(temp_output, fieldnames=temp_fieldnames, 
                                   quotechar='"', quoting=csv.QUOTE_MINIMAL)
        temp_writer.writeheader()

        def get_key_values(row):
            """
            行から複数のキー値を取得し、タプルとして返します。
            """
            return tuple(row[col] for col in key_columns)

        def handle_new_columns():
            """
            新しい列が追加された場合に一時ファイルを再構成します。
            """
            nonlocal temp_writer, temp_fieldnames
            temp_fieldnames = list(original_fieldnames) + sorted(list(current_fieldnames - set(original_fieldnames)))
            
            # 既存のデータを保存
            existing_content = []
            with open(temp_output_path, 'r', newline='', encoding='cp932') as old_temp:
                old_reader = csv.DictReader(old_temp, quotechar='"')
                for row in old_reader:
                    existing_content.append(row)
            
            # 新しい列を含めて一時ファイルを書き直し
            with open(temp_output_path, 'w', newline='', encoding='cp932') as new_temp:
                new_writer = csv.DictWriter(new_temp, fieldnames=temp_fieldnames,
                                          quotechar='"', quoting=csv.QUOTE_MINIMAL)
                new_writer.writeheader()
                for row in existing_content:
                    new_writer.writerow(row)
            
            # 書き込みオブジェクトを更新
            temp_writer = csv.DictWriter(temp_output, fieldnames=temp_fieldnames,
                                       quotechar='"', quoting=csv.QUOTE_MINIMAL)

        def merge_group(group):
            """
            同じキー値を持つ行のグループをマージします。
            """
            if not group:
                return None
            
            merged_row = {}
            column_values = {}
            
            # グループ内の全ての値を収集
            for row in group:
                for col in row.keys():
                    if col not in column_values:
                        column_values[col] = [row[col]]
                    else:
                        column_values[col].append(row[col])

            # 各列の値を処理
            new_columns_added = False
            for col, values in column_values.items():
                unique_values = []
                for val in values:
                    if val not in unique_values:
                        unique_values.append(val)
                
                if len(unique_values) == 1:
                    merged_row[col] = unique_values[0]
                else:
                    for idx, val in enumerate(unique_values):
                        col_name = col if idx == 0 else f"{col}_{idx}"
                        merged_row[col_name] = val
                        if col_name not in current_fieldnames:
                            current_fieldnames.add(col_name)
                            new_columns_added = True
            
            if new_columns_added:
                handle_new_columns()
            
            return merged_row

        # メインの処理ループ
        for row in reader:
            key_values = get_key_values(row)
            if key_values == current_key_values or current_key_values is None:
                current_group.append(row)
                current_key_values = key_values
            else:
                merged_row = merge_group(current_group)
                if merged_row:
                    temp_writer.writerow(merged_row)
                current_group = [row]
                current_key_values = key_values

        # 最後のグループを処理
        if current_group:
            merged_row = merge_group(current_group)
            if merged_row:
                temp_writer.writerow(merged_row)

    # 最終的な出力ファイルの作成
    final_fieldnames = list(original_fieldnames) + sorted(list(current_fieldnames - set(original_fieldnames)))
    
    with open(temp_output_path, 'r', newline='', encoding='cp932') as temp_input, \
         open(csv_output_path, 'w', newline='', encoding='cp932') as csvfile_out:
        
        reader = csv.DictReader(temp_input, quotechar='"')
        writer = csv.DictWriter(csvfile_out, fieldnames=final_fieldnames,
                              quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        
        for row in reader:
            complete_row = {field: row.get(field, '') for field in final_fieldnames}
            writer.writerow(complete_row)

    # 一時ファイルの削除
    try:
        os.remove(temp_output_path)
    except:
        pass

# 使用例：
# csv_input_path = 'input.csv'
# csv_output_path = 'output.csv'
# key_columns = ['id', 'category']  # 複数のキー列を指定
# merge_consecutive_rows(csv_input_path, csv_output_path, key_columns)
