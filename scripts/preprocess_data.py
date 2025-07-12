
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Charger les données
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('/home/legbedje/Documents/datascienceproject/predictionimmobilier/data/raw/train.csv', header=None, delimiter=',', names=column_names)

# Séparer les caractéristiques et la variable cible
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mettre à l'échelle les caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Enregistrer les données prétraitées
processed_data_path = '/home/legbedje/Documents/datascienceproject/predictionimmobilier/data/processed/'
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(processed_data_path + 'X_train.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(processed_data_path + 'X_test.csv', index=False)
y_train.to_csv(processed_data_path + 'y_train.csv', index=False)
y_test.to_csv(processed_data_path + 'y_test.csv', index=False)

print("Les données ont été prétraitées et enregistrées dans le répertoire data/processed.")
