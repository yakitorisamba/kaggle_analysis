import pandas as pd
import os
import shutil

def remove_columns(input_dir, output_dir, list_file_path):
    """
    CSVファイルから不要な列を削除する
    """
    # リストファイルの読み込み
    list_df = pd.read_csv(list_file_path)
    list_df.columns = ['filename', 'column_name', 'is_needed']
    
    # 必要な列の辞書を作成
    needed_columns = {}
    for fname in list_df['filename'].unique():
        file_rows = list_df[list_df['filename'] == fname]
        needed_cols = file_rows[file_rows['is_needed']]['column_name'].tolist()
        needed_columns[fname] = needed_cols
    
    # 一時的な出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in needed_columns.keys():
        input_file_path = os.path.join(input_dir, filename)
        if os.path.exists(input_file_path):
            # ファイルの読み込みと列の削除
            df = pd.read_csv(input_file_path)
            df = df[needed_columns[filename]]
            
            # 結果の保存
            output_file_path = os.path.join(output_dir, filename)
            df.to_csv(output_file_path, index=False)
            print(f"Processed and saved: {filename}")
        else:
            print(f"File not found: {filename}")

def append_data(new_result_dir, existing_result_dir):
    """
    既存の結果に新しいデータを追加し、既存のフォルダを更新する
    """
    # 一時ディレクトリの作成
    temp_dir = "temp_combined_results"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 新しい結果ディレクトリ内のファイルを処理
    for filename in os.listdir(new_result_dir):
        if not filename.endswith('.csv'):
            continue
            
        new_file_path = os.path.join(new_result_dir, filename)
        existing_file_path = os.path.join(existing_result_dir, filename)
        temp_file_path = os.path.join(temp_dir, filename)
        
        # 新しいデータの読み込み
        new_df = pd.read_csv(new_file_path)
        
        try:
            # 既存データの読み込みと結合
            if os.path.exists(existing_file_path):
                existing_df = pd.read_csv(existing_file_path)
                
                # 列名の確認
                if set(existing_df.columns) == set(new_df.columns):
                    # データの結合
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    
                    # 重複行の削除（必要な場合）
                    combined_df = combined_df.drop_duplicates()
                    
                    # 一時ファイルとして保存
                    combined_df.to_csv(temp_file_path, index=False)
                    print(f"Combined: {filename}")
                else:
                    print(f"Column mismatch in {filename} - skipping")
                    continue
            else:
                # 既存ファイルが無い場合は新しいデータをそのまま保存
                new_df.to_csv(temp_file_path, index=False)
                print(f"Created new file: {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    try:
        # 処理が成功したら、既存のファイルを更新
        for filename in os.listdir(temp_dir):
            temp_file_path = os.path.join(temp_dir, filename)
            target_file_path = os.path.join(existing_result_dir, filename)
            
            # 既存のファイルを更新
            shutil.copy2(temp_file_path, target_file_path)
            print(f"Updated: {filename}")
            
        # 一時ディレクトリの削除
        shutil.rmtree(temp_dir)
        print("Successfully updated all files")
        
    except Exception as e:
        print(f"Error during file update: {e}")
        print("Temporary files are kept in: {temp_dir}")

if __name__ == "__main__":
    # 設定
    list_file_path = "list.csv"
    input_dir = "input"
    new_result_dir = "result"
    existing_result_dir = "existing_results"
    
    # 列の削除が必要な場合のみ実行
    # remove_columns(input_dir, "filtered_results", list_file_path)
    
    # データの追加（既存フォルダを更新）
    append_data(new_result_dir, existing_result_dir)
