from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd
import numpy as np
import glob
import time
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Lendo dataset do script principal
dados_filtrados_com_linha_100 = pd.read_csv("./content/dataset.csv")
dados = dados_filtrados_com_linha_100.groupby("DataHoraPartida120", as_index=False)['KmPercorridos'].sum().set_index('DataHoraPartida120')
dados = dados.asfreq(freq='D')
# Aplicando ARIMA
model = ARIMA(dados, order=(1,1,1))
res = model.fit()
print(res.summary())
# Aplicando SARIMA
results = sm.tsa.statespace.SARIMAX(dados, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=0)
print(results.summary())
# Recolhendo forecasts ARIMA
forecasts = res.forecast(steps=60, alpha=0.05)
# Recolhendo forecasts SARIMA
forecast_values = results.get_forecast(steps=60)
forecast_ci = forecast_values.conf_int()
forecast_values = results.get_forecast(steps=60)
# Recolhendo metricas de erro do ARIMA
mse = mean_squared_error(np.array(dados['KmPercorridos'])[-60:], np.array(forecasts))
mae = mean_absolute_error(np.array(dados['KmPercorridos'])[-60:], np.array(forecasts))
rmse = mse**.5
print("MSE: ", mse)
print("MAE: ",mae)
print("RMSE: ", rmse)
# Recolhendo metricas de erro do SARIMA
mse = mean_squared_error(np.array(dados['KmPercorridos'])[-60:], np.array(forecast_values.predicted_mean))
mae = mean_absolute_error(np.array(dados['KmPercorridos'])[-60:], np.array(forecast_values.predicted_mean))
rmse = mse**.5
print("MSE: ", mse)
print("MAE: ",mae)
print("RMSE: ", rmse)
# Aplicando Cross Validation ARIMA
cross_validation = dados
tscv = TimeSeriesSplit(max_train_size=150 ,n_splits = 10)
arima_rmse = []
arima_mae = []
arima_mse = []
forecasts = []
for train_index, test_index in tscv.split(cross_validation):
    cv_train, cv_test = cross_validation.iloc[train_index], cross_validation.iloc[test_index]
    
    arima = sm.tsa.ARIMA(cv_train, order=(1,1,1)).fit()

    forecasts.append(arima.forecast(steps=40))
    
    predictions = arima.predict(cv_test.index.values[0], cv_test.index.values[-1])
    true_values = cv_test.values
    arima_rmse.append(sqrt(mean_squared_error(true_values, predictions)))
    arima_mse.append(mean_squared_error(true_values, predictions))
    arima_mae.append(mean_absolute_error(true_values, predictions))
    
print("RMSE: {}".format(np.mean(arima_rmse)))
print("MAE: {}".format(np.mean(arima_mae)))
print("MSE: {}".format(np.mean(arima_mse)))
print(forecasts[-1:])
# Aplicando Cross Validation SARIMA
cross_validation = dados
tscv = TimeSeriesSplit(max_train_size=150 ,n_splits = 10)
sarima_rmse = []
sarima_mae = []
sarima_mse = []
forecasts = []
for train_index, test_index in tscv.split(cross_validation):
    cv_train, cv_test = cross_validation.iloc[train_index], cross_validation.iloc[test_index]
    
    sarima =  sm.tsa.statespace.SARIMAX(cv_train, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=0)

    forecasts.append(sarima.forecast(steps=40))
    
    predictions = sarima.predict(cv_test.index.values[0], cv_test.index.values[-1])
    true_values = cv_test.values
    sarima_rmse.append(sqrt(mean_squared_error(true_values, predictions)))
    sarima_mse.append(mean_squared_error(true_values, predictions))
    sarima_mae.append(mean_absolute_error(true_values, predictions))
    
print("RMSE: {}".format(np.mean(sarima_rmse)))
print("MAE: {}".format(np.mean(sarima_mae)))
print("MSE: {}".format(np.mean(sarima_mse)))
print(forecasts[-1:])
