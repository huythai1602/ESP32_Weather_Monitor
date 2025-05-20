# models/training/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

# Đường dẫn đến file dữ liệu
DATA_PATH = "../../data/processed/processed_data.csv"  # Đường dẫn đến dữ liệu đã xử lý
MODELS_DIR = "../saved_models"
COEF_DIR = "../coefficients"

def train_models(prediction_horizon=6):
    """Huấn luyện mô hình với dữ liệu đã xử lý"""
    # Tạo thư mục cho models nếu chưa tồn tại
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(COEF_DIR, exist_ok=True)
    
    # Đọc dữ liệu đã xử lý
    print(f"Đang đọc dữ liệu từ {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Kiểm tra và hiển thị thông tin dữ liệu ban đầu
    print(f"Dữ liệu ban đầu có {len(df)} mẫu")
    print("5 dòng đầu tiên:")
    print(df.head())
    print("\nThống kê mô tả:")
    print(df.describe())

    # Kiểm tra dữ liệu
    print("\nKiểm tra dữ liệu:")
    print("Số lượng giá trị NaN:")
    print(df.isna().sum())
    
    # Kiểm tra số lượng dữ liệu
    print(f"Số lượng bản ghi: {len(df)}")
    if len(df) < 30:
        print("CẢNH BÁO: Dữ liệu quá ít cho mô hình chính xác!")
        print("Đang thực hiện data augmentation...")
        # Thực hiện data augmentation đơn giản
        original_df = df.copy()
        for i in range(5):  # Tạo thêm 5 bản sao với nhiễu nhỏ
            noisy_df = original_df.copy()
            for col in ['temp_dht', 'hum_dht']:
                if col in noisy_df.columns:
                    # Thêm nhiễu ngẫu nhiên khoảng ±5%
                    noise = np.random.normal(0, 0.05 * noisy_df[col].std(), size=len(noisy_df))
                    noisy_df[col] = noisy_df[col] + noise
            df = pd.concat([df, noisy_df])
        print(f"Đã tăng kích thước dữ liệu lên {len(df)} bản ghi")
    
    # Chuẩn bị đặc trưng cho mô hình
    # Lọc các cột số (ngoại trừ timestamp và các cột mục tiêu)
    feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Loại bỏ các cột mục tiêu và timestamp
    for col in ['timestamp', 'temp_dht', 'hum_dht']:
        if col in feature_cols:
            feature_cols.remove(col)
    
    print(f"Các đặc trưng được sử dụng: {feature_cols}")
    
    # Tạo đặc trưng X và mục tiêu y
    X = df[feature_cols].values
    
    # Tạo nhãn cho nhiệt độ và độ ẩm trong tương lai (sau n bản ghi)
    y_temp = df['temp_dht'].shift(-prediction_horizon).values
    y_hum = df['hum_dht'].shift(-prediction_horizon).values
    
    # Xóa các hàng cuối không có nhãn
    mask = ~np.isnan(y_temp)
    X = X[mask]
    y_temp = y_temp[mask]
    y_hum = y_hum[mask]
    
    print(f"Kích thước đặc trưng cuối cùng: {X.shape}")
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Đã chuẩn hóa dữ liệu - trung bình 0, phương sai 1")
    
    # Chọn đặc trưng tốt nhất nếu có quá nhiều đặc trưng so với số mẫu
    if X.shape[1] > X.shape[0] / 5:  # Nếu số đặc trưng > 20% số mẫu
        print("Số lượng đặc trưng nhiều so với số mẫu, thực hiện lựa chọn đặc trưng...")
        k = min(5, X_scaled.shape[1])  # Chọn tối đa 5 đặc trưng hoặc ít hơn
        selector = SelectKBest(f_regression, k=k)
        X_selected = selector.fit_transform(X_scaled, y_temp)
        
        # Lấy chỉ số của các đặc trưng được chọn
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_cols[i] for i in selected_indices]
        print(f"Đặc trưng được chọn: {selected_features}")
        
        # Cập nhật X_scaled và feature_cols
        X_scaled = X_selected
        feature_cols = selected_features
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_temp_train, y_temp_test = train_test_split(
        X_scaled, y_temp, test_size=0.2, random_state=42)
    _, _, y_hum_train, y_hum_test = train_test_split(
        X_scaled, y_hum, test_size=0.2, random_state=42, shuffle=False)
    
    # Huấn luyện mô hình nhiệt độ
    print("\nĐang huấn luyện mô hình nhiệt độ...")
    temp_model = LinearRegression()
    temp_model.fit(X_train, y_temp_train)
    
    # Đánh giá mô hình nhiệt độ
    temp_pred = temp_model.predict(X_test)
    temp_rmse = np.sqrt(mean_squared_error(y_temp_test, temp_pred))
    temp_r2 = r2_score(y_temp_test, temp_pred)
    
    print(f"Mô hình nhiệt độ - RMSE: {temp_rmse:.2f}°C, R²: {temp_r2:.2f}")
    
    # Cross-validation
    cv_size = min(5, len(X_scaled))
    if cv_size <= 1:
        print("CẢNH BÁO: Không đủ dữ liệu cho cross-validation")
    else:
        temp_cv_scores = cross_val_score(LinearRegression(), X_scaled, y_temp, 
                                       cv=cv_size, scoring='r2')
        print(f"Điểm R² cross-validation (nhiệt độ): {np.mean(temp_cv_scores):.2f} ± {np.std(temp_cv_scores):.2f}")
    
    # Huấn luyện mô hình độ ẩm
    print("\nĐang huấn luyện mô hình độ ẩm...")
    hum_model = LinearRegression()
    hum_model.fit(X_train, y_hum_train)
    
    # Đánh giá mô hình độ ẩm
    hum_pred = hum_model.predict(X_test)
    hum_rmse = np.sqrt(mean_squared_error(y_hum_test, hum_pred))
    hum_r2 = r2_score(y_hum_test, hum_pred)
    
    print(f"Mô hình độ ẩm - RMSE: {hum_rmse:.2f}%, R²: {hum_r2:.2f}")
    
    # Cross-validation
    if cv_size > 1:
        hum_cv_scores = cross_val_score(LinearRegression(), X_scaled, y_hum, 
                                       cv=cv_size, scoring='r2')
        print(f"Điểm R² cross-validation (độ ẩm): {np.mean(hum_cv_scores):.2f} ± {np.std(hum_cv_scores):.2f}")
    
    # Trực quan hóa kết quả
    plt.figure(figsize=(12, 8))
    
    # Biểu đồ nhiệt độ
    plt.subplot(2, 2, 1)
    plt.scatter(y_temp_test, temp_pred, alpha=0.5)
    plt.plot([min(y_temp_test), max(y_temp_test)], [min(y_temp_test), max(y_temp_test)], 'r--')
    plt.title(f'Nhiệt độ: Thực tế vs Dự đoán (R² = {temp_r2:.2f})')
    plt.xlabel('Thực tế (°C)')
    plt.ylabel('Dự đoán (°C)')
    
    # Biểu đồ độ ẩm
    plt.subplot(2, 2, 2)
    plt.scatter(y_hum_test, hum_pred, alpha=0.5)
    plt.plot([min(y_hum_test), max(y_hum_test)], [min(y_hum_test), max(y_hum_test)], 'r--')
    plt.title(f'Độ ẩm: Thực tế vs Dự đoán (R² = {hum_r2:.2f})')
    plt.xlabel('Thực tế (%)')
    plt.ylabel('Dự đoán (%)')
    
    # Biểu đồ nhiệt độ theo thời gian
    plt.subplot(2, 2, 3)
    plt.plot(range(len(y_temp_test)), y_temp_test, 'b-', label='Thực tế')
    plt.plot(range(len(temp_pred)), temp_pred, 'r--', label='Dự đoán')
    plt.title('Nhiệt độ theo thời gian')
    plt.xlabel('Mẫu')
    plt.ylabel('Nhiệt độ (°C)')
    plt.legend()
    
    # Biểu đồ độ ẩm theo thời gian
    plt.subplot(2, 2, 4)
    plt.plot(range(len(y_hum_test)), y_hum_test, 'b-', label='Thực tế')
    plt.plot(range(len(hum_pred)), hum_pred, 'r--', label='Dự đoán')
    plt.title('Độ ẩm theo thời gian')
    plt.xlabel('Mẫu')
    plt.ylabel('Độ ẩm (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("../../docs/images/model_evaluation.png")
    
    # Kiểm tra các hệ số bất thường
    def check_large_coefficients(model, feature_names, model_name):
        large_coefs = []
        for i, coef in enumerate(model.coef_):
            if abs(coef) > 100:  # Hệ số lớn bất thường
                large_coefs.append((feature_names[i], coef))
        
        if large_coefs:
            print(f"\nCẢNH BÁO: Phát hiện hệ số lớn bất thường trong mô hình {model_name}:")
            for feature, coef in large_coefs:
                print(f"  - {feature}: {coef:.6f}")
            print("  Điều này có thể dẫn đến dự đoán không ổn định!")
    
    # Kiểm tra hệ số lớn
    check_large_coefficients(temp_model, feature_cols, "nhiệt độ")
    check_large_coefficients(hum_model, feature_cols, "độ ẩm")
    
    # Lưu mô hình
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Lưu mô hình bằng pickle
    temp_model_path = f"{MODELS_DIR}/temp_model_{timestamp}.pkl"
    hum_model_path = f"{MODELS_DIR}/hum_model_{timestamp}.pkl"
    scaler_path = f"{MODELS_DIR}/scaler_{timestamp}.pkl"
    
    with open(temp_model_path, 'wb') as f:
        pickle.dump(temp_model, f)
    
    with open(hum_model_path, 'wb') as f:
        pickle.dump(hum_model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nĐã lưu mô hình nhiệt độ vào: {temp_model_path}")
    print(f"Đã lưu mô hình độ ẩm vào: {hum_model_path}")
    print(f"Đã lưu scaler vào: {scaler_path}")
    
    # Lưu hệ số cho ESP32
    coef_file = f"{COEF_DIR}/model_coef_{timestamp}.h"
    
    with open(coef_file, 'w', encoding='utf-8') as f:
        f.write("// Hệ số mô hình dự đoán thời tiết\n")
        f.write("// Được tạo tự động bởi script train_model.py\n")
        f.write(f"// Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"// Dựa trên {len(X)} mẫu dữ liệu\n\n")
        
        f.write("#ifndef MODEL_COEF_H\n")
        f.write("#define MODEL_COEF_H\n\n")
        
        # Lưu thông tin chuẩn hóa
        f.write("// Thông tin chuẩn hóa dữ liệu\n")
        f.write("const float feature_means[] = {")
        
        # Nếu đã chọn đặc trưng, chỉ lưu giá trị means cho những đặc trưng được chọn
        if 'selected_indices' in locals():
            for i, idx in enumerate(selected_indices):
                f.write(f"{scaler.mean_[idx]:.6f}f")
                if i < len(selected_indices) - 1:
                    f.write(", ")
        else:
            for i, mean in enumerate(scaler.mean_):
                f.write(f"{mean:.6f}f")
                if i < len(scaler.mean_) - 1:
                    f.write(", ")
        f.write("};\n\n")
        
        f.write("const float feature_scales[] = {")
        if 'selected_indices' in locals():
            for i, idx in enumerate(selected_indices):
                f.write(f"{scaler.scale_[idx]:.6f}f")
                if i < len(selected_indices) - 1:
                    f.write(", ")
        else:
            for i, scale in enumerate(scaler.scale_):
                f.write(f"{scale:.6f}f")
                if i < len(scaler.scale_) - 1:
                    f.write(", ")
        f.write("};\n\n")
        
        f.write("// Hệ số cho mô hình nhiệt độ\n")
        f.write(f"const float temp_intercept = {temp_model.intercept_:.6f}f;\n")
        f.write("const float temp_coef[] = {")
        for i, coef in enumerate(temp_model.coef_):
            f.write(f"{coef:.6f}f")
            if i < len(temp_model.coef_) - 1:
                f.write(", ")
        f.write("};\n\n")
        
        f.write("// Hệ số cho mô hình độ ẩm\n")
        f.write(f"const float hum_intercept = {hum_model.intercept_:.6f}f;\n")
        f.write("const float hum_coef[] = {")
        for i, coef in enumerate(hum_model.coef_):
            f.write(f"{coef:.6f}f")
            if i < len(hum_model.coef_) - 1:
                f.write(", ")
        f.write("};\n\n")
        
        f.write("// Thứ tự các đặc trưng\n")
        f.write("// ")
        for i, feature in enumerate(feature_cols):
            f.write(feature)
            if i < len(feature_cols) - 1:
                f.write(", ")
        f.write("\n\n")
        
        f.write("// Số lượng đặc trưng\n")
        f.write(f"const int NUM_FEATURES = {len(feature_cols)};\n\n")
        
        f.write("#endif // MODEL_COEF_H\n")
    
    print(f"Đã lưu hệ số mô hình vào: {coef_file}")
    print("\nQuá trình huấn luyện mô hình hoàn tất!")
    
    # Hiển thị đóng góp của các đặc trưng
    print("\n=== ĐÓNG GÓP CỦA CÁC ĐẶC TRƯNG ===")
    
    # Tính toán tầm quan trọng tương đối
    temp_importance = np.abs(temp_model.coef_) / np.sum(np.abs(temp_model.coef_)) * 100
    hum_importance = np.abs(hum_model.coef_) / np.sum(np.abs(hum_model.coef_)) * 100
    
    # In tầm quan trọng của từng đặc trưng
    print("\nĐộ quan trọng của các đặc trưng cho mô hình nhiệt độ:")
    for i, feature in enumerate(feature_cols):
        print(f"{feature}: {temp_importance[i]:.2f}% (hệ số: {temp_model.coef_[i]:.4f})")
    
    print("\nĐộ quan trọng của các đặc trưng cho mô hình độ ẩm:")
    for i, feature in enumerate(feature_cols):
        print(f"{feature}: {hum_importance[i]:.2f}% (hệ số: {hum_model.coef_[i]:.4f})")
    
    # Tạo mô hình Fall-back nếu phát hiện hệ số bất thường
    has_large_coefficients = False
    for coef in temp_model.coef_:
        if abs(coef) > 100:
            has_large_coefficients = True
            break
    
    if not has_large_coefficients:
        for coef in hum_model.coef_:
            if abs(coef) > 100:
                has_large_coefficients = True
                break
    
    if has_large_coefficients:
        print("\n⚠️ Phát hiện hệ số bất thường, tạo mô hình đơn giản fall-back...")
        fallback_file = f"{COEF_DIR}/model_coef_fallback_{timestamp}.h"
        
        with open(fallback_file, 'w', encoding='utf-8') as f:
            f.write("// Hệ số mô hình dự đoán thời tiết đơn giản (fall-back)\n")
            f.write("// Được tạo tự động bởi script train_model.py\n")
            f.write(f"// Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"// Dựa trên {len(X)} mẫu dữ liệu\n\n")
            
            f.write("#ifndef MODEL_COEF_H\n")
            f.write("#define MODEL_COEF_H\n\n")
            
            f.write("// Hệ số đơn giản cho mô hình nhiệt độ - Weighted Average\n")
            f.write("const float temp_intercept = 0.0f;\n")
            
            # Tạo hệ số đơn giản
            simple_temp_coef = ["0.0f"] * len(feature_cols)
            simple_hum_coef = ["0.0f"] * len(feature_cols)
            
            # Gán trọng số cho các đặc trưng có tên phù hợp
            for i, feature in enumerate(feature_cols):
                if "temp_api" in feature:
                    simple_temp_coef[i] = "0.6f"
                elif "temp_dht_lag1" in feature or "temp_dht_prev" in feature:
                    simple_temp_coef[i] = "0.3f"
                elif "temp_dht_diff" in feature:
                    simple_temp_coef[i] = "0.1f"
                
                if "hum_api" in feature:
                    simple_hum_coef[i] = "0.6f"
                elif "hum_dht_lag1" in feature or "hum_dht_prev" in feature:
                    simple_hum_coef[i] = "0.3f"
                elif "hum_dht_diff" in feature:
                    simple_hum_coef[i] = "0.1f"
            
            f.write("const float temp_coef[] = {" + ", ".join(simple_temp_coef) + "};\n\n")
            
            f.write("// Hệ số đơn giản cho mô hình độ ẩm - Weighted Average\n")
            f.write("const float hum_intercept = 0.0f;\n")
            f.write("const float hum_coef[] = {" + ", ".join(simple_hum_coef) + "};\n\n")
            
            # Không cần chuẩn hóa dữ liệu cho mô hình đơn giản
            f.write("// Giá trị trung bình và tỷ lệ cho chuẩn hóa - giá trị 0 và 1 để không thực hiện chuẩn hóa\n")
            f.write("const float feature_means[] = {" + ", ".join(["0.0f"] * len(feature_cols)) + "};\n")
            f.write("const float feature_scales[] = {" + ", ".join(["1.0f"] * len(feature_cols)) + "};\n\n")
            
            f.write("// Thứ tự các đặc trưng\n")
            f.write("// ")
            for i, feature in enumerate(feature_cols):
                f.write(feature)
                if i < len(feature_cols) - 1:
                    f.write(", ")
            f.write("\n\n")
            
            f.write("// Số lượng đặc trưng\n")
            f.write(f"const int NUM_FEATURES = {len(feature_cols)};\n\n")
            
            f.write("#endif // MODEL_COEF_H\n")
        
        print(f"Đã lưu mô hình fall-back vào: {fallback_file}")
        print("LƯU Ý: Đề xuất sử dụng mô hình fall-back này nếu mô hình chính không ổn định!")
    
    return temp_model, hum_model, feature_cols

if __name__ == "__main__":
    temp_model, hum_model, features = train_models()
    plt.show()