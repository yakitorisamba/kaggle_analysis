import csv
import os

def merge_consecutive_rows(csv_input_path, csv_output_path, key_column):
    temp_output_path = csv_output_path + '.tmp'
    current_fieldnames = set()
    
    # First pass to get initial fieldnames
    with open(csv_input_path, 'r', newline='', encoding='cp932') as csvfile_in:
        reader = csv.DictReader(csvfile_in, quotechar='"')
        original_fieldnames = reader.fieldnames.copy()
        current_fieldnames.update(original_fieldnames)

    with open(csv_input_path, 'r', newline='', encoding='cp932') as csvfile_in, \
         open(temp_output_path, 'w', newline='', encoding='cp932') as temp_output:
        
        reader = csv.DictReader(csvfile_in, quotechar='"')
        temp_writer = None  # Will be initialized after we know all fieldnames
        written_header = False
        
        current_group = []
        current_key_value = None

        def merge_group(group):
            if not group:
                return None
            merged_row = {}
            column_values = {}
            
            # Collect all values for each column across the group
            for row in group:
                for col in row.keys():  # Use row.keys() to catch any new columns
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

        def write_row(row_data, writer):
            nonlocal written_header, temp_writer
            
            # Check if we have new columns
            new_columns = set(row_data.keys()) - current_fieldnames
            if new_columns:
                # Update fieldnames set
                current_fieldnames.update(new_columns)
                
                # Create new writer with updated fieldnames
                fieldnames_list = list(original_fieldnames) + sorted(list(current_fieldnames - set(original_fieldnames)))
                temp_writer = csv.DictWriter(temp_output, fieldnames=fieldnames_list, quotechar='"', quoting=csv.QUOTE_MINIMAL)
                
                if not written_header:
                    temp_writer.writeheader()
                    written_header = True
            
            # If writer hasn't been initialized yet, initialize it
            if temp_writer is None:
                fieldnames_list = list(original_fieldnames)
                temp_writer = csv.DictWriter(temp_output, fieldnames=fieldnames_list, quotechar='"', quoting=csv.QUOTE_MINIMAL)
                if not written_header:
                    temp_writer.writeheader()
                    written_header = True
            
            temp_writer.writerow(row_data)

        # Process rows
        for row in reader:
            # Add any new columns to current_fieldnames
            current_fieldnames.update(row.keys())
            
            key_value = row[key_column]
            if key_value == current_key_value or current_key_value is None:
                current_group.append(row)
                current_key_value = key_value
            else:
                merged_row = merge_group(current_group)
                if merged_row:
                    write_row(merged_row, temp_writer)
                current_group = [row]
                current_key_value = key_value

        # Process the last group
        if current_group:
            merged_row = merge_group(current_group)
            if merged_row:
                write_row(merged_row, temp_writer)

    # Write final output file
    final_fieldnames = list(original_fieldnames) + sorted(list(current_fieldnames - set(original_fieldnames)))
    
    with open(temp_output_path, 'r', newline='', encoding='cp932') as temp_input, \
         open(csv_output_path, 'w', newline='', encoding='cp932') as csvfile_out:
        
        reader = csv.DictReader(temp_input, quotechar='"')
        writer = csv.DictWriter(csvfile_out, fieldnames=final_fieldnames, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        
        for row in reader:
            writer.writerow(row)

    # Clean up temporary file
    os.remove(temp_output_path)

# Usage example:
# csv_input_path = 'input.csv'
# csv_output_path = 'output.csv'
# key_column = 'id'
# merge_consecutive_rows(csv_input_path, csv_output_path, key_column)
