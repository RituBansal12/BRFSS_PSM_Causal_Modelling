import pyreadstat
import pandas as pd
import os

# List files in the data directory for debugging
print('Files in data/:', os.listdir('data'))

# Note: The filename has a trailing space, so we must include it
input_path = 'data/LLCP2023.XPT '
try:
    df, meta = pyreadstat.read_xport(input_path, encoding='latin1')
except UnicodeDecodeError:
    print('latin1 encoding failed, trying cp1252...')
    df, meta = pyreadstat.read_xport(input_path, encoding='cp1252')

# Save as CSV
output_path = 'data/LLCP2023.csv'
df.to_csv(output_path, index=False)

print(f"Converted {input_path} to {output_path}")
