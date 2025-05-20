// Hệ số mô hình dự đoán thời tiết
// Được tạo tự động bởi script train_model.py
// Thời gian: 2025-05-20 23:43:57
// Dựa trên 57 mẫu dữ liệu

#ifndef MODEL_COEF_H
#define MODEL_COEF_H

// Thông tin chuẩn hóa dữ liệu
const float feature_means[] = {24.491053f, 51.561404f, 30.891354f, 30.907143f, 65.973504f, 64.868241f, 0.008772f, 0.526316f, 6.409073f, 14.938416f};

const float feature_scales[] = {13.474799f, 29.238894f, 4.762683f, 4.766533f, 15.712398f, 15.630693f, 3.877714f, 12.870603f, 11.945424f, 30.679985f};

// Hệ số cho mô hình nhiệt độ
const float temp_intercept = 31.122623f;
const float temp_coef[] = {-1.929946f, 2.942981f, -4.597228f, 5.165839f, 3.112453f, -6.107360f, -2.259184f, 1.491176f, -0.389268f, -0.585172f};

// Hệ số cho mô hình độ ẩm
const float hum_intercept = 67.452068f;
const float hum_coef[] = {-1.346722f, 3.747488f, 6.598737f, -1.833567f, -3.051731f, 1.439381f, -5.109145f, 6.422393f, 2.491557f, -2.440104f};

// Thứ tự các đặc trưng
// temp_api, hum_api, temp_dht_lag1, temp_dht_lag3, hum_dht_lag1, hum_dht_lag3, temp_dht_diff, hum_dht_diff, temp_diff_dht_api, hum_diff_dht_api

// Số lượng đặc trưng
const int NUM_FEATURES = 10;

#endif // MODEL_COEF_H
