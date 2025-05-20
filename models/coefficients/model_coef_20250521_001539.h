// Hệ số mô hình dự đoán thời tiết
// Được tạo tự động bởi script train_model.py
// Thời gian: 2025-05-21 00:15:39
// Dựa trên 64 mẫu dữ liệu

#ifndef MODEL_COEF_H
#define MODEL_COEF_H

// Thông tin chuẩn hóa dữ liệu
const float feature_means[] = {30.792187f, 70.671875f, 31.904687f, 31.950000f, 69.265625f, 68.640625f, -0.015625f, 0.312500f, 1.096875f, -1.093750f};

const float feature_scales[] = {2.810520f, 11.357892f, 0.737909f, 0.725862f, 10.987553f, 10.870498f, 0.600057f, 3.353520f, 2.564197f, 3.706914f};

// Hệ số cho mô hình nhiệt độ
const float temp_intercept = 31.839192f;
const float temp_coef[] = {-1.948487f, -2.111169f, -0.701454f, 0.032233f, -1.716938f, -0.524559f, -0.434950f, -0.648794f, 1.832019f, 0.792501f};

// Hệ số cho mô hình độ ẩm
const float hum_intercept = 69.380103f;
const float hum_coef[] = {7.358917f, 5.988835f, 0.570912f, 2.156788f, 4.229673f, 5.290446f, -0.215926f, 5.091701f, -7.952069f, -1.206302f};

// Thứ tự các đặc trưng
// temp_api, hum_api, temp_dht_lag1, temp_dht_lag3, hum_dht_lag1, hum_dht_lag3, temp_dht_diff, hum_dht_diff, temp_diff_dht_api, hum_diff_dht_api

// Số lượng đặc trưng
const int NUM_FEATURES = 10;

#endif // MODEL_COEF_H
