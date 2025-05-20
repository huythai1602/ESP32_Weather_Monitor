// models/coefficients/model_coef.h
#ifndef MODEL_COEF_H
#define MODEL_COEF_H

// Hệ số cho mô hình nhiệt độ
const float temp_intercept = 5.23;
const float temp_coef[] = {-0.12, 0.03, 0.85, -0.02};

// Hệ số cho mô hình độ ẩm
const float hum_intercept = 12.45;
const float hum_coef[] = {0.5, -0.1, -0.15, 0.92};

#endif