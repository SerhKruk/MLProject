import pandas as pd

df = pd.read_csv('BankChurners.csv')

from sklearn.model_selection import train_test_split

training_data, testing_data = train_test_split(df, test_size=0.3)

training_data.to_csv('train.csv', index=False)
testing_data.to_csv('val.csv', index=False)
print("\n".join(training_data.columns))