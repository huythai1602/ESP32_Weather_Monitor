# models/training/explore_data.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Đường dẫn đến file dữ liệu
DATA_PATH = "../../data/raw/weather_data.csv"  # Điều chỉnh đường dẫn nếu cần

def explore_data():
    # Tạo thư mục cho báo cáo nếu chưa tồn tại
    os.makedirs("../../docs/images", exist_ok=True)
    
    # Đọc dữ liệu
    print(f"Đang đọc dữ liệu từ {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Hiển thị thông tin cơ bản
    print("\n=== THÔNG TIN DỮ LIỆU ===")
    print(f"Tổng số bản ghi: {len(df)}")
    print("\nCác cột trong dữ liệu:")
    print(df.columns.tolist())
    
    print("\nMẫu dữ liệu:")
    print(df.head())
    
    print("\nThống kê mô tả:")
    print(df.describe())
    
    # Kiểm tra giá trị thiếu
    print("\nGiá trị thiếu:")
    print(df.isnull().sum())
    
    # Chuyển đổi timestamp sang datetime nếu có
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print("\nThời gian bắt đầu:", df['timestamp'].min())
        print("Thời gian kết thúc:", df['timestamp'].max())
        print(f"Tổng thời gian: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600:.1f} giờ")
    
    # Trực quan hóa
    print("\nĐang tạo biểu đồ phân tích...")
    
    # Biểu đồ nhiệt độ và độ ẩm theo thời gian
    if 'timestamp' in df.columns and 'temp_dht' in df.columns and 'hum_dht' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(df['timestamp'], df['temp_dht'], 'r-', label='DHT')
        if 'temp_api' in df.columns:
            plt.plot(df['timestamp'], df['temp_api'], 'b--', label='API')
        plt.title('Nhiệt độ theo thời gian')
        plt.ylabel('Nhiệt độ (°C)')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(df['timestamp'], df['hum_dht'], 'g-', label='DHT')
        if 'hum_api' in df.columns:
            plt.plot(df['timestamp'], df['hum_api'], 'm--', label='API')
        plt.title('Độ ẩm theo thời gian')
        plt.ylabel('Độ ẩm (%)')
        plt.xlabel('Thời gian')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("../../docs/images/time_series_plot.png")
    
    # Biểu đồ phân phối
    if 'temp_dht' in df.columns and 'hum_dht' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(df['temp_dht'], kde=True)
        plt.title('Phân phối nhiệt độ (DHT)')
        plt.xlabel('Nhiệt độ (°C)')
        
        plt.subplot(1, 2, 2)
        sns.histplot(df['hum_dht'], kde=True)
        plt.title('Phân phối độ ẩm (DHT)')
        plt.xlabel('Độ ẩm (%)')
        
        plt.tight_layout()
        plt.savefig("../../docs/images/distribution_plot.png")
    
    # Biểu đồ tương quan
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Ma trận tương quan giữa các biến')
        plt.tight_layout()
        plt.savefig("../../docs/images/correlation_matrix.png")
    
    print(f"Các biểu đồ đã được lưu trong thư mục docs/images")
    print("Khám phá dữ liệu hoàn tất!")
    
    return df

if __name__ == "__main__":
    df = explore_data()
    plt.show()  # Hiển thị tất cả biểu đồ