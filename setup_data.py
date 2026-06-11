import os
import urllib.request
import pandas as pd

def download_hotel_data():
    """
    Downloads the Hotel Booking Demand dataset and saves it to the data/ directory.
    """
    os.makedirs('data', exist_ok=True)
    file_path = 'data/hotel_bookings.csv'
    
    if os.path.exists(file_path):
        print(f"Data already exists at {file_path}")
        return
        
    print("Downloading Hotel Booking Demand dataset...")
    # Reliable open source link for the Kaggle hotel booking dataset
    url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
    
    try:
        urllib.request.urlretrieve(url, file_path)
        print(f"Successfully downloaded to {file_path}")
        
        # Quick check
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        print("Data is ready!")
    except Exception as e:
        print(f"Failed to download data: {e}")

if __name__ == "__main__":
    download_hotel_data()
