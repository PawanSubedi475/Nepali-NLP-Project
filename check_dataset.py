import pandas as pd

df = pd.read_csv('nepali_news_dataset/train.csv')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'\nCategories distribution:')
print(df['category'].value_counts())
print(f'\nFirst row sample:')
print(df.iloc[0])
