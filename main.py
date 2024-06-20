import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings
import ray

# Ignore warnings
warnings.filterwarnings("ignore")

# Initialize Ray
ray.init()

# Load dataset
data = pd.read_csv('time_series_sku_dataset_new.csv')

# Ensure the Date column is of datetime type
data['Date'] = pd.to_datetime(data['Date'])


# Simple preprocessing function
def preprocess_data(data):
    # Fill missing values with median
    data['Sales'].fillna(data['Sales'].median(), inplace=True)
    return data

# Preprocess the entire dataset
data = preprocess_data(data)

@ray.remote
def arima_forecast(train, test):
    """
    Perform ARIMA forecasting on the given training and testing data.

    Parameters:
    train (pd.Series): Training data series.
    test (pd.Series): Testing data series.

    Returns:
    list: Predictions for the testing data.
    """
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        try:
            model = ARIMA(history, order=(5,1,0))  # You can tune the (p,d,q) parameters
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test.iloc[t])
        except Exception as e:
            print(f"ARIMA model fitting failed: {e}")
            predictions.append(np.nan)
    return predictions

@ray.remote
def prophet_forecast(train, test):
    """
    Perform Prophet forecasting on the given training and testing data.

    Parameters:
    train (pd.Series): Training data series.
    test (pd.Series): Testing data series.

    Returns:
    np.ndarray: Predictions for the testing data.
    """
    df_train = pd.DataFrame({'ds': train.index, 'y': train.values})
    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=len(test), freq='D')
    forecast = model.predict(future)
    predictions = forecast['yhat'][-len(test):].values
    return predictions

def rmse(actual, predicted):
    """
    Calculate the Root Mean Squared Error (RMSE) between actual and predicted values.

    Parameters:
    actual (np.ndarray): Actual values.
    predicted (np.ndarray): Predicted values.

    Returns:
    float: RMSE value.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    actual.setflags(write=True)
    predicted.setflags(write=True)
    return np.sqrt(mean_squared_error(actual, predicted))

@ray.remote
def evaluate_models(train, test):
    """
    Evaluate ARIMA and Prophet models on the given training and testing data.

    Parameters:
    train (pd.Series): Training data series.
    test (pd.Series): Testing data series.

    Returns:
    tuple: Best model name, RMSE value, and predictions.
    """
    arima_preds = ray.get(arima_forecast.remote(train, test))
    prophet_preds = ray.get(prophet_forecast.remote(train, test))
    
    arima_rmse = rmse(test, arima_preds)
    prophet_rmse = rmse(test, prophet_preds)
    print('ARIMA', arima_rmse, arima_preds)
    print('Prophet', prophet_rmse, prophet_preds)
    
    if arima_rmse < prophet_rmse:
        return 'ARIMA', arima_rmse, arima_preds
    else:
        return 'Prophet', prophet_rmse, prophet_preds

@ray.remote
def forecast_for_sku(sku_data):
    """
    Forecast sales for a specific SKU using ARIMA and Prophet models.

    Parameters:
    sku_data (pd.DataFrame): Data for a specific SKU.

    Returns:
    tuple: SKU identifier and forecast results.
    """
    if len(sku_data) < 10:  # Ensure enough data points
        return None, f"Skipping SKU due to insufficient data."

    train_size = int(len(sku_data) * 0.8)
    train, test = sku_data['Sales'][:train_size], sku_data['Sales'][train_size:]
    train.index = sku_data['Date'][:train_size]
    test.index = sku_data['Date'][train_size:]

    best_model, rmse_value, predictions = ray.get(evaluate_models.remote(train, test))
    return sku_data['SKU'].iloc[0], {
        'best_model': best_model,
        'rmse': rmse_value,
        'predictions': predictions,
        'dates': test.index,
        'actual': test.values
    }

def forecast_all_skus(data):
    """
    Forecast sales for all SKUs in the dataset.

    Parameters:
    data (pd.DataFrame): The entire dataset.

    Returns:
    pd.DataFrame: DataFrame containing forecast results for all SKUs.
    """
    skus = data['SKU'].unique()
    results = []

    # Create a list of Ray tasks
    tasks = [forecast_for_sku.remote(data[data['SKU'] == sku].sort_values('Date')) for sku in skus]

    # Execute tasks in parallel
    results_list = ray.get(tasks)

    # Collect results
    for sku, result in results_list:
        if sku is not None:
            for date, actual, pred in zip(result['dates'], result['actual'], result['predictions']):
                results.append({
                    'SKU': sku,
                    'Date': date,
                    'Actual Sales': actual,
                    'Predicted Sales': pred,
                    'Best Model': result['best_model'],
                    'RMSE': result['rmse']
                })

    return pd.DataFrame(results)

# Forecast for all SKUs and store results in a DataFrame
results_df = forecast_all_skus(data)

# Save the results to a CSV file
results_df.to_csv('sku_predictions.csv', index=False)

print("Predictions saved to 'sku_predictions.csv'")
