import pandas as pd
import os

file_path = 'data/hotel_bookings.csv'
if not os.path.exists(file_path):
    print("hotel_bookings.csv not found")
    exit(1)

df = pd.read_csv(file_path)

# Handle Missing Values
df['company'] = df['company'].fillna(0)
df['agent'] = df['agent'].fillna(0)
df['country'] = df['country'].fillna('Unknown')
df['children'] = df['children'].fillna(0)

# Outliers
df = df[(df['adr'] >= 0) & (df['adr'] < 5000)]

# Feature Engineering
df['total_duration'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df = df[df['total_duration'] > 0]

df.to_csv('data/cleaned_hotel_bookings.csv', index=False)
print("Cleaned data generated at data/cleaned_hotel_bookings.csv")
