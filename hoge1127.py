import csv
import os

def merge_consecutive_rows(csv_input_path, csv_output_path, key_column):
    temp_output_path = csv_output_path + '.tmp'
    all_fieldnames = set()

    # First pass: collect all possible fieldnames
    with open(csv_input_path, 'r', newline='', encoding='utf-8') as csvfile_in:
        reader = csv.DictReader(csvfile_in)
        original_fieldnames = reader.fieldnames.copy()
        all_fieldnames.update(original_fieldnames)
        key_index = original_fieldnames.index(key_column)

    with open(csv_input_path, 'r', newline='', encoding='utf-8') as csvfile_in, \
         open(temp_output_path, 'w', newline='', encoding='utf-8') as temp_output:

        reader = csv.DictReader(csvfile_in)
        current_group = []
        current_key_value = None

        # Function to merge a group of rows
        def merge_group(group):
            if not group:
                return None
            merged_row = {}
            column_values = {}
            
            # Collect all values for each column
            for row in group:
                for col in original_fieldnames:
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
                        all_fieldnames.add(col_name)
            
            return merged_row

        # Initialize writer with all known fieldnames
        temp_writer = csv.DictWriter(temp_output, fieldnames=original_fieldnames, extrasaction='ignore')
        temp_writer.writeheader()

        # Process rows
        for row in reader:
            key_value = row[key_column]
            if key_value == current_key_value or current_key_value is None:
                current_group.append(row)
                current_key_value = key_value
            else:
                merged_row = merge_group(current_group)
                if merged_row:
                    temp_writer.writerow(merged_row)
                current_group = [row]
                current_key_value = key_value

        # Process the last group
        if current_group:
            merged_row = merge_group(current_group)
            if merged_row:
                temp_writer.writerow(merged_row)

    # Prepare final fieldnames list
    final_fieldnames = list(original_fieldnames)
    extra_fieldnames = sorted(list(all_fieldnames - set(original_fieldnames)))
    final_fieldnames.extend(extra_fieldnames)

    # Write final output with all columns
    with open(temp_output_path, 'r', newline='', encoding='utf-8') as temp_input, \
         open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile_out:

        reader = csv.DictReader(temp_input)
        writer = csv.DictWriter(csvfile_out, fieldnames=final_fieldnames)
        writer.writeheader()

        for row in reader:
            # Create a new row with all fields initialized to empty string
            new_row = {field: '' for field in final_fieldnames}
            # Update with actual values
            new_row.update(row)
            writer.writerow(new_row)

    # Clean up temporary file
    os.remove(temp_output_path)

# Usage example:
# csv_input_path = 'large_file.csv'   # Replace with your input CSV file path
# csv_output_path = 'merged_output.csv'  # Replace with your desired output CSV file path
# key_column = 'your_key_column'         # Replace with your key column name
# merge_consecutive_rows(csv_input_path, csv_output_path, key_column)
