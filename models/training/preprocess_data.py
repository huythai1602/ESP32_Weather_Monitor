# models/training/preprocess_data.py
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Đường dẫn đến file dữ liệu
DATA_PATH = "../../data/raw/weather_data.csv"  # Điều chỉnh đường dẫn nếu cần
PROCESSED_PATH = "../../data/processed/processed_data.csv"

def preprocess_data():
    # Tạo thư mục cho dữ liệu đã xử lý nếu chưa tồn tại
    os.makedirs("../../data/processed", exist_ok=True)
    
    # Đọc dữ liệu
    print(f"Đang đọc dữ liệu từ {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Hiển thị thông tin dữ liệu ban đầu
    print(f"Dữ liệu ban đầu có {len(df)} bản ghi và {len(df.columns)} cột")
    print("Các cột trong dữ liệu: ", df.columns.tolist())
    
    # Kiểm tra dữ liệu thiếu và giá trị 0
    print("\nSố lượng giá trị N/A trong mỗi cột:")
    print(df.isna().sum())
    
    # Đếm số bản ghi có giá trị 0 trong các cột số
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    print("\nSố lượng giá trị 0 trong mỗi cột số:")
    for col in numeric_cols:
        zero_count = (df[col] == 0).sum()
        print(f"{col}: {zero_count} giá trị 0")
    
    # Xóa các bản ghi có giá trị 0 hoặc N/A trong các cột quan trọng
    important_cols = ['temp_dht', 'hum_dht', 'temp_api', 'hum_api']
    original_count = len(df)
    
    # Lọc ra các bản ghi không có giá trị 0 trong các cột quan trọng
    for col in important_cols:
        if col in df.columns:
            df = df[df[col] != 0]
    
    # Xóa các bản ghi có giá trị N/A
    df = df.dropna()
    
    zero_na_removed = original_count - len(df)
    print(f"\nĐã xóa {zero_na_removed} bản ghi có giá trị 0 hoặc N/A")
    print(f"Còn lại {len(df)} bản ghi")
    
    # Chuyển đổi timestamp sang datetime nếu có
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Đã chuyển đổi cột timestamp sang datetime")
    
    # Thêm các đặc trưng thời gian
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        print(f"Đã thêm các đặc trưng thời gian: hour, day_of_week, day_of_year")
    
    # Xử lý giá trị ngoại lệ
    for col in ['temp_dht', 'hum_dht', 'temp_api', 'hum_api']:
        if col in df.columns:
            # Loại bỏ giá trị ngoại lệ (nằm ngoài 3 độ lệch chuẩn)
            mean = df[col].mean()
            std = df[col].std()
            outliers_before = ((df[col] < mean - 3*std) | (df[col] > mean + 3*std)).sum()
            df[col] = df[col].clip(mean - 3*std, mean + 3*std)
            print(f"Đã xử lý {outliers_before} giá trị ngoại lệ cho cột {col}")
    
    # Thêm các đặc trưng lag (dữ liệu trước đó)
    for col in ['temp_dht', 'hum_dht']:
        if col in df.columns:
            df[f'{col}_lag1'] = df[col].shift(1)  # Giá trị trước đó 1 step
            df[f'{col}_lag3'] = df[col].shift(3)  # Giá trị trước đó 3 step
            print(f"Đã thêm đặc trưng độ trễ cho cột {col}")
    
    # Thêm đặc trưng biến thiên
    for col in ['temp_dht', 'hum_dht']:
        if col in df.columns and f'{col}_lag1' in df.columns:
            df[f'{col}_diff'] = df[col] - df[f'{col}_lag1']  # Sự thay đổi
            print(f"Đã thêm đặc trưng biến thiên cho cột {col}")
    
    # Thêm đặc trưng chênh lệch giữa DHT và API (nếu có)
    if 'temp_dht' in df.columns and 'temp_api' in df.columns:
        df['temp_diff_dht_api'] = df['temp_dht'] - df['temp_api']
        print("Đã thêm đặc trưng chênh lệch nhiệt độ giữa DHT và API")
    
    if 'hum_dht' in df.columns and 'hum_api' in df.columns:
        df['hum_diff_dht_api'] = df['hum_dht'] - df['hum_api']
        print("Đã thêm đặc trưng chênh lệch độ ẩm giữa DHT và API")
    
    # Xóa các bản ghi có giá trị thiếu sau khi tạo đặc trưng mới
    rows_before = len(df)
    df = df.dropna()
    na_removed = rows_before - len(df)
    print(f"Đã xóa {na_removed} hàng có giá trị thiếu sau khi tạo đặc trưng mới")
    
    # Kiểm tra nếu còn giá trị 0 trong các đặc trưng tạo ra
    derived_features = [col for col in df.columns if '_lag' in col or '_diff' in col]
    for col in derived_features:
        zero_count = (df[col] == 0).sum()
        print(f"{col}: {zero_count} giá trị 0")
    
    # Lưu dữ liệu đã xử lý
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Đã lưu dữ liệu đã xử lý vào {PROCESSED_PATH}")
    print(f"Số bản ghi cuối cùng: {len(df)}")
    
    # Hiển thị thống kê mô tả
    print("\nThống kê mô tả của dữ liệu sau xử lý:")
    print(df.describe())
    
    return df

if __name__ == "__main__":
    df = preprocess_data()