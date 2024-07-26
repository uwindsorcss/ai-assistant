"""Script to combine all CSV files into one corpus"""
import pandas as pd
import os
import csv

# Get all CSV files in the data directory
input_files = ["./data/base/" + file for file in os.listdir("./data/base/") if file.endswith('.csv')]

# Write to a unified "corpus.csv" file
output_file = "./data/corpus.csv"

try:
    # Read and concatenate all CSV files
    dataframes = [pd.read_csv(file) for file in input_files]
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Write the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
    print(f"Successfully concatenated {len(input_files)} files into {output_file}")
except Exception as e:
    print(f"An error occurred: {e}")
