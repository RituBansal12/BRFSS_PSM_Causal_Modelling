import pyreadstat
import pandas as pd
import os
import re
import html

# List files in the data directory for debugging
print('Files in data/:', os.listdir('data'))

# Note: The filename has a trailing space, so we must include it
input_path = 'data/LLCP2023.XPT '
try:
    df, meta = pyreadstat.read_xport(input_path, encoding='latin1')
except UnicodeDecodeError:
    print('latin1 encoding failed, trying cp1252...')
    df, meta = pyreadstat.read_xport(input_path, encoding='cp1252')

# Read the HTML file
with open('data/USCODE23_LLCP_021924.HTML', 'r', encoding='windows-1252') as f:
    raw_html = f.read()

# Extract "Label", "Section Name", and "SAS Variable Name"
pattern = re.compile(
    r'Label:&nbsp;(.+?)<br>.*?Section&nbsp;Name:&nbsp;(.+?)<br>.*?SAS&nbsp;Variable&nbsp;Name:&nbsp;(.+?)<br>',
    re.DOTALL
)
matches = pattern.findall(raw_html)

# Clean and create DataFrame
clean = [
    (
        html.unescape(var.replace('\xa0', ' ')).strip(),
        html.unescape(label.replace('\xa0', ' ')).strip(),
        html.unescape(section.replace('\xa0', ' ')).strip()
    )
    for label, section, var in matches
]
df_column_mapping = pd.DataFrame(clean, columns=['SAS Variable Name', 'Label', 'Section Name'])

# Extract variable blocks with labels and HTML table contents
question_blocks = re.findall(
    r'Label:&nbsp;(.+?)<br>.*?SAS&nbsp;Variable&nbsp;Name:&nbsp;(.+?)<br>(.*?)(?=<b>|Label:&nbsp;|</body>)',
    raw_html,
    re.DOTALL
)

results = []

for label, varname, block in question_blocks:
    # Look for value-label pairs in <td> rows
    rows = re.findall(r'<tr>\s*<td[^>]*>(.*?)</td>\s*<td[^>]*>(.*?)</td>', block, re.DOTALL)
    for value, value_label in rows:
        clean_value = html.unescape(re.sub(r'<.*?>', '', value)).strip()
        clean_label = html.unescape(re.sub(r'<.*?>', '', value_label)).strip()
        if clean_value and clean_label:
            results.append((varname.strip(), clean_value, clean_label))

# Convert to DataFrame
df_values = pd.DataFrame(results, columns=["SAS Variable Name", "Value", "Value Label"])
df_values = df_values.drop_duplicates()

# Merge with column mapping
df_column_mapping = df_column_mapping.merge(df_values, on='SAS Variable Name', how='left')

# Save as CSV
output_path = 'data/LLCP2023.csv'
df.to_csv(output_path, index=False)

# Save the column mapping DataFrame
df_column_mapping.to_csv('data/LLCP2023_metadata.csv', index=False)

print(f"Converted {input_path} to {output_path}")
