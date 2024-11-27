import csv
import os

def merge_consecutive_rows(csv_input_path, csv_output_path, key_columns):
    # キー列が文字列として渡された場合、リストに変換します
    if isinstance(key_columns, str):
        key_columns = [key_columns]
    
    temp_output_path = csv_output_path + '.tmp'
    current_fieldnames = set()
    all_rows = []
    
    # 最初にキー列の存在確認とフィールド名の取得を行います
    with open(csv_input_path, 'r', newline='', encoding='cp932') as csvfile_in:
        reader = csv.DictReader(csvfile_in, quotechar='"')
        # キー列の存在確認
        missing_columns = [col for col in key_columns if col not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"指定されたキー列が見つかりません: {', '.join(missing_columns)}")
        original_fieldnames = reader.fieldnames.copy()
        current_fieldnames.update(original_fieldnames)

    with open(csv_input_path, 'r', newline='', encoding='cp932') as csvfile_in:
        reader = csv.DictReader(csvfile_in, quotechar='"')
        current_group = []
        current_key_values = None

        def get_key_values(row):
            # 行から複数のキー値をタプルとして取得します
            return tuple(row[col] for col in key_columns)

        def merge_group(group):
            if not group:
                return None
            merged_row = {}
            column_values = {}
            
            # グループ内の全ての値を収集します
            for row in group:
                for col in row.keys():
                    if col not in column_values:
                        column_values[col] = [row[col]]
                    else:
                        column_values[col].append(row[col])

            # 各列を処理します
            for col, values in column_values.items():
                # 重複を除いた値のリストを作成
                unique_values = []
                for val in values:
                    if val not in unique_values:
                        unique_values.append(val)
                
                if len(unique_values) == 1:
                    # 値が1つの場合はそのまま使用
                    merged_row[col] = unique_values[0]
                else:
                    # 複数の値がある場合は新しい列を作成
                    for idx, val in enumerate(unique_values):
                        col_name = col if idx == 0 else f"{col}_{idx}"
                        merged_row[col_name] = val
                        current_fieldnames.add(col_name)
            
            return merged_row

        # 行の処理
        for row in reader:
            # 複数のキー値を取得
            key_values = get_key_values(row)
            if key_values == current_key_values or current_key_values is None:
                # 同じキー値の場合はグループに追加
                current_group.append(row)
                current_key_values = key_values
            else:
                # キー値が変わった場合は現在のグループをマージ
                merged_row = merge_group(current_group)
                if merged_row:
                    all_rows.append(merged_row)
                current_group = [row]
                current_key_values = key_values

        # 最後のグループを処理
        if current_group:
            merged_row = merge_group(current_group)
            if merged_row:
                all_rows.append(merged_row)

    # 最終的な列名リストを作成（元の列順を維持しつつ、新しい列を追加）
    final_fieldnames = list(original_fieldnames) + sorted(list(current_fieldnames - set(original_fieldnames)))
    
    # 最終的な出力ファイルを作成
    with open(csv_output_path, 'w', newline='', encoding='cp932') as csvfile_out:
        writer = csv.DictWriter(csvfile_out, fieldnames=final_fieldnames,
                              quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        
        # 全ての行を書き込み
        for row in all_rows:
            # 全てのフィールドが存在することを確認
            complete_row = {field: row.get(field, '') for field in final_fieldnames}
            writer.writerow(complete_row)

# 使用例:
# csv_input_path = 'input.csv'
# csv_output_path = 'output.csv'
# key_columns = ['id', 'category']  # 複数のキー列を指定
# merge_consecutive_rows(csv_input_path, csv_output_path, key_columns)
