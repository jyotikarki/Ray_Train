import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sku_codes(num_skus):
    """
    Generate a list of SKU codes.

    Parameters:
    num_skus (int): Number of unique SKUs to generate.

    Returns:
    list: List of SKU codes.
    """
    return ['SKU{:04d}'.format(i) for i in range(1, num_skus + 1)]

def generate_random_data(sku_codes, products, date_range):
    """
    Generate random data for the dataset.

    Parameters:
    sku_codes (list): List of SKU codes.
    products (list): List of product names.
    date_range (pd.DatetimeIndex): Range of dates for the data.

    Returns:
    pd.DataFrame: DataFrame containing the generated dataset.
    """
    data = pd.DataFrame(columns=['Date', 'SKU', 'Product', 'Sales', 'Price'])
    
    for sku in sku_codes:
        product = np.random.choice(products)
        sales = np.random.randint(10, 100, len(date_range))
        price = np.random.uniform(50, 1000, len(date_range)).round(2)
        
        sku_data = pd.DataFrame({
            'Date': date_range,
            'SKU': sku,
            'Product': product,
            'Sales': sales,
            'Price': price
        })
        
        data = pd.concat([data, sku_data])

    return data

def save_to_csv(data, filename):
    """
    Save DataFrame to a CSV file.

    Parameters:
    data (pd.DataFrame): DataFrame to be saved.
    filename (str): Name of the file to save the DataFrame to.
    """
    data.to_csv(filename, index=False)
    print(f"Time series SKU dataset saved to '{filename}' with {len(data)} rows.")

def main():
    """
    Main function to generate and save the time series SKU dataset.
    """
    # Define parameters
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    num_skus = 10  # Number of unique SKUs
    date_range = pd.date_range(start=start_date, end=end_date)

    # Generate SKU codes and random data
    sku_codes = generate_sku_codes(num_skus)
    products = ['Smartphone', 'Laptop', 'TV', 'Headphones', 'Jacket', 'Dress', 'Refrigerator', 'Microwave', 'Book', 'Football', 'Doll']
    data = generate_random_data(sku_codes, products, date_range)

    # Save DataFrame to CSV (optional)
    save_to_csv(data, 'time_series_sku_dataset_new.csv')

if __name__ == "__main__":
    main()
