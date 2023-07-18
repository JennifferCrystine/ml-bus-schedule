from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np


#lendo do dataset limpo
dataset = pd.read_csv('./content/dataset.csv')

#transformando atributos discretos em contínuos
label_encoder = LabelEncoder()

#separação de atributos categoricos para codificar
categorical_attbr = ['DataHoraPartida120', 'dia_semana_partida']

#atributos numéricos
numerical_attbr = dataset.drop('DataHoraPartida120', axis=1)
numerical_attbr = numerical_attbr.drop('dia_semana_partida', axis=1)

X_categorical = dataset[categorical_attbr]
X_encoded = pd.DataFrame()

#codificação de valores categoricos
for column in categorical_attbr:
    encoded_column = label_encoder.fit_transform(X_categorical[column])
    X_encoded[column] = encoded_column

#concatena valores categoricos codificado com continuos
X = pd.concat([numerical_attbr, X_encoded], axis=1)
y = dataset['KmPercorridos']

#remoção de sujeira
X =  X.drop('Unnamed: 0', axis=1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

#configurando semente do gerador de números aleatórios
np.random.seed(20)

rf = RandomForestRegressor()

#dicionário com valores de hiperparâmetros para serem variados
param_grid = {
  "n_estimators": [200, 300, 400],
  "criterion": ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
  "max_features": ["sqrt", "log2"],
  "max_depth": [3, 5, 7],
  "min_samples_split": [100, 200, 300], #100,
  "bootstrap": [True, False],
  "min_samples_leaf": [100, 200]
}

grid = GridSearchCV(rf, param_grid, cv=10, scoring='neg_mean_squared_error', verbose=1)

grid_search=grid.fit(X, y)



np.random.seed(20)

# {'bootstrap': False, 'criterion': 'poisson', 'max_depth': 5, 'max_features': 'log2', 'min_samples_leaf': 100, 'min_samples_split': 200, 'n_estimators': 300}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestRegressor(n_estimators=300,
                           max_depth=5,
                           max_features= 'log2',
                           min_samples_leaf=100,
                           bootstrap=False,
                           criterion='poisson',
                           min_samples_split=200
                           )

scores_list = ['neg_root_mean_squared_error', 'neg_mean_squared_error', 'neg_mean_absolute_error']
scores = cross_validate(rf, X, y, scoring=scores_list, cv=10)

#treinando modelo com conjunto de testes
rf.fit(X_train, y_train)
y_prediction = rf.predict(X_test)

mse_scores = -scores['test_neg_mean_squared_error']
rmse_scores = -scores['test_neg_root_mean_squared_error']
mae_scores = -scores['test_neg_mean_absolute_error']
confidence_rmse = 1.96 * np.std(rmse_scores) / np.sqrt(len(rmse_scores))
confidence_mse = 1.96 * np.std(mse_scores) / np.sqrt(len(mse_scores))
confidence_mae = 1.96 * np.std(mae_scores) / np.sqrt(len(mae_scores))


print('Médias \nMSE {0} \nRMSE {1} \nMAE {2}'.format(-1 * scores['test_neg_mean_squared_error'].mean(),
                                                      -1 * scores['test_neg_root_mean_squared_error'].mean(),
                                                      -1 * scores['test_neg_mean_absolute_error'].mean()
                                                      ))


mse = mean_squared_error(y_test, y_prediction)
mae = mean_absolute_error(y_test, y_prediction)
rmse = mse**.5
print("MSE: ", mse)
print("MAE: ",mae)
print("RMSE: ", rmse)