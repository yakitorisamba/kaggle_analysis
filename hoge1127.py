import csv
import os

def merge_consecutive_rows(csv_input_path, csv_output_path, key_column):
    temp_output_path = csv_output_path + '.tmp'
    fieldnames_set = set()

    with open(csv_input_path, 'r', newline='', encoding='utf-8') as csvfile_in, \
         open(temp_output_path, 'w', newline='', encoding='utf-8') as temp_output:

        reader = csv.DictReader(csvfile_in)
        original_fieldnames = reader.fieldnames.copy()
        key_index = original_fieldnames.index(key_column)

        current_group = []
        current_key_value = None
        temp_fieldnames_set = set(original_fieldnames)

        # Function to merge a group of rows
        def merge_group(group):
            if not group:
                return None
            merged_row = {}
            column_values = {}
            for row in group:
                for col in original_fieldnames:
                    if col not in column_values:
                        column_values[col] = [row[col]]
                    else:
                        column_values[col].append(row[col])

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
                        temp_fieldnames_set.add(col_name)
            return merged_row

        temp_writer = csv.DictWriter(temp_output, fieldnames=original_fieldnames)
        temp_writer.writeheader()

        for row in reader:
            key_value = row[key_column]
            if key_value == current_key_value or current_key_value is None:
                current_group.append(row)
                current_key_value = key_value
            else:
                merged_row = merge_group(current_group)
                temp_writer.writerow(merged_row)
                current_group = [row]
                current_key_value = key_value
        if current_group:
            merged_row = merge_group(current_group)
            temp_writer.writerow(merged_row)

    # Now we have temp_fieldnames_set containing all fieldnames
    # Write the final output file
    temp_fieldnames = list(original_fieldnames)
    extra_fieldnames = list(temp_fieldnames_set - set(original_fieldnames))
    temp_fieldnames.extend(extra_fieldnames)

    with open(temp_output_path, 'r', newline='', encoding='utf-8') as temp_output, \
         open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile_out:

        reader = csv.DictReader(temp_output)
        writer = csv.DictWriter(csvfile_out, fieldnames=temp_fieldnames)
        writer.writeheader()
        for row in reader:
            writer.writerow(row)

    os.remove(temp_output_path)

# Usage example:
csv_input_path = 'large_file.csv'   # Replace with your input CSV file path
csv_output_path = 'merged_output.csv'  # Replace with your desired output CSV file path
key_column = 'your_key_column'         # Replace with your key column name

merge_consecutive_rows(csv_input_path, csv_output_path, key_column)
