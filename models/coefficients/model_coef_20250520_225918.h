// Hệ số mô hình dự đoán thời tiết
// Được tạo tự động bởi script train_model.py
// Thời gian: 2025-05-20 22:59:18
// Dựa trên 29 mẫu dữ liệu

#ifndef MODEL_COEF_H
#define MODEL_COEF_H

// Hệ số cho mô hình nhiệt độ
const float temp_intercept = 12195.904165f;
const float temp_coef[] = {114352728848.156891f, 18041624621.719837f, -114352729092.655670f, 0.133631f, -18041624688.151413f, -0.026731f, -114352729092.140747f, -18041624688.232845f, 114352729092.138046f, 18041624688.190414f};

// Hệ số cho mô hình độ ẩm
const float hum_intercept = 35765.186928f;
const float hum_coef[] = {3998700531155.970703f, 630881787966.963623f, -3998700531874.270996f, 0.314324f, -630881788162.474854f, 0.814819f, -3998700531875.725098f, -630881788160.756592f, 3998700531871.348633f, 630881788161.814575f};

// Thứ tự các đặc trưng
// temp_api, hum_api, temp_dht_lag1, temp_dht_lag3, hum_dht_lag1, hum_dht_lag3, temp_dht_diff, hum_dht_diff, temp_diff_dht_api, hum_diff_dht_api

// Số lượng đặc trưng
const int NUM_FEATURES = 10;

#endif // MODEL_COEF_H
