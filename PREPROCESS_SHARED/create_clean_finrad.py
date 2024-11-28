import pandas as pd

#define path
path = "your_path"

df = pd.read_csv(f'{path}/DATA_SHARED/FinRAD_13K_terms_definitions_labels.csv')
df['sentence'] = df['terms'] + ' is ' + df['definitions']
df[['sentence']].to_csv(f'{path}/DATA_SHARED/finrad_clean.csv')