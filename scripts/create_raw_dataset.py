import pandas as pd

df = pd.read_excel('../../MT700Data.xlsx', sheet_name='in')

df = df[[':45A:']]

df.to_csv('data/raw_dataset.csv', index=False, header=False)
