import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('/home/legbedje/Documents/datascienceproject/predictionimmobilier/data/raw/train.csv', header=None, delimiter=',', names=column_names)

print("--- df.head() ---")
print(df.head())
print("\n--- df.info() ---")
df.info()
print("\n--- df.describe() ---")
print(df.describe())
print("\n--- Correlation Matrix ---")
print(df.corr())