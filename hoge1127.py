import csv
import os

def merge_consecutive_rows(csv_input_path, csv_output_path, key_column):
    temp_output_path = csv_output_path + '.tmp'
    current_fieldnames = set()
    all_rows = []
    
    # First pass to get initial fieldnames
    with open(csv_input_path, 'r', newline='', encoding='cp932') as csvfile_in:
        reader = csv.DictReader(csvfile_in, quotechar='"')
        original_fieldnames = reader.fieldnames.copy()
        current_fieldnames.update(original_fieldnames)

    with open(csv_input_path, 'r', newline='', encoding='cp932') as csvfile_in:
        reader = csv.DictReader(csvfile_in, quotechar='"')
        current_group = []
        current_key_value = None

        def merge_group(group):
            if not group:
                return None
            merged_row = {}
            column_values = {}
            
            # Collect all values for each column across the group
            for row in group:
                for col in row.keys():
                    if col not in column_values:
                        column_values[col] = [row[col]]
                    else:
                        column_values[col].append(row[col])

            # Process each column
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
                        current_fieldnames.add(col_name)
            
            return merged_row

        # Process rows
        for row in reader:
            key_value = row[key_column]
            if key_value == current_key_value or current_key_value is None:
                current_group.append(row)
                current_key_value = key_value
            else:
                merged_row = merge_group(current_group)
                if merged_row:
                    all_rows.append(merged_row)
                current_group = [row]
                current_key_value = key_value

        # Process the last group
        if current_group:
            merged_row = merge_group(current_group)
            if merged_row:
                all_rows.append(merged_row)

    # Create final output with all columns
    final_fieldnames = list(original_fieldnames) + sorted(list(current_fieldnames - set(original_fieldnames)))
    
    # Write final output file
    with open(csv_output_path, 'w', newline='', encoding='cp932') as csvfile_out:
        writer = csv.DictWriter(csvfile_out, fieldnames=final_fieldnames,
                              quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        
        for row in all_rows:
            # Ensure all fields are present in the row
            complete_row = {field: row.get(field, '') for field in final_fieldnames}
            writer.writerow(complete_row)

# Usage example:
# csv_input_path = 'input.csv'
# csv_output_path = 'output.csv'
# key_column = 'id'
# merge_consecutive_rows(csv_input_path, csv_output_path, key_column)
