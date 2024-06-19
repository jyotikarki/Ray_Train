# Time Series Forecasting Using Multiple Models per SKU

This repository contains scripts for generating synthetic time series data, training forecasting models using ARIMA and Prophet, selecting the best model per SKU, and generating predictions.

## Dataset Generation

1. **Setup**
   - Ensure Python 3.x is installed.
   - Install required packages:
     ```bash
     pip install pandas numpy
     ```

2. **Generate Dataset**
   - Modify parameters in `dataset.py` (e.g., `start_date`, `end_date`, `num_skus`, `products`).
   - Run the script to generate the dataset:
     ```bash
     python dataset.py
     ```
   - This will create a CSV file `time_series_sku_dataset_new.csv` with synthetic data.

## Model Training and Evaluation

1. **Setup**
   - Install additional dependencies:
     ```bash
     pip install statsmodels prophet scikit-learn ray
     ```

2. **Train Models**
   - Modify parameters in `main.py` if necessary (e.g., file paths).
   - Execute the script to train models and generate forecasts:
     ```bash
     python main.py
     ```
   - Models (ARIMA and Prophet) will be trained using parallel processing with Ray.

3. **View Results**
   - The predictions for each SKU will be saved in `sku_predictions.csv`.
   - Check the file for forecasted sales and model performance metrics (RMSE).

## Additional Notes

- **Python Environment**: Consider using a virtual environment (`venv` or `conda`) for package management.
- **Customization**: Adjust model parameters (`order` for ARIMA, forecasting settings for Prophet) based on specific requirements.
- **Deployment**: Adapt scripts for different datasets or integrate into larger forecasting pipelines.

### Structure Explanation:
- **Introduction**: Briefly introduces the purpose of the repository.
- **Dataset Generation**: Instructions on setting up and generating synthetic data using dataset.py.
  - Provides steps to modify parameters and run the script.
- **Model Training and Evaluation**: Details on setting up dependencies, training models with main.py, and viewing forecast results.
- Includes notes on customization and deployment.
- **Additional Notes**: Provides tips on managing Python environments, customizing scripts, and potential extensions.


