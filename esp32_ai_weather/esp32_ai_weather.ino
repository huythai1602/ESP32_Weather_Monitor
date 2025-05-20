/*
 * Hệ Thống Giám Sát và Dự Báo Môi Trường Thông Minh Dựa trên IoT và Trí Tuệ Nhân Tạo
 * (IoT-based Smart Environmental Monitoring and Prediction System with Artificial Intelligence)
 * 
 * Mô tả:
 * - Sử dụng ESP32 và DHT11 để đọc nhiệt độ và độ ẩm
 * - Lấy dữ liệu dự báo từ OpenWeather API
 * - Dự đoán nhiệt độ và độ ẩm trong tương lai bằng mô hình AI
 * - Cảnh báo bằng đèn LED khi dự đoán vượt ngưỡng
 * - Lưu trữ dữ liệu trên Google Sheets
 */

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "DHT.h"
#include "time.h"
#include "model_coef.h"  // File header chứa hệ số mô hình AI

// Định nghĩa chân kết nối
#define DHTPIN 23     // Chân kết nối DHT11
#define DHTTYPE DHT11
#define LED_PIN 2     // Chân kết nối LED

// Ngưỡng cảnh báo
#define TEMP_THRESHOLD 27.0  // Ngưỡng nhiệt độ (27°C)
#define HUM_THRESHOLD 30.0   // Ngưỡng độ ẩm (30%)

// Thông tin WiFi
const char* ssid = "HUY_THAI";
const char* password = "Huythai2003";

// Thông tin NTP Server để lấy thời gian
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 7 * 3600; // GMT+7 (Việt Nam)
const int daylightOffset_sec = 0;

// OpenWeather API
const char* apiKey = "ae437f23c1997a658605be5235d3d7e2";
const char* city = "Hanoi";
const char* country = "VN";
const char* units = "metric";

// Google Sheets API
const char* scriptUrl = "https://script.google.com/macros/s/AKfycbz8PNvMcZhqdUdaxWj70VaNGPWxSBVm7Mjy8tR4bnS7IsYa7bN47Or58cw_LgGqFf_EpQ/exec";

// Biến toàn cục
float last_t_dht = 0;
float last_h_dht = 0;
float last_t_api = 0;
float last_h_api = 0;
float predicted_temp = 0;
float predicted_hum = 0;
float prev_t_dht = 0;  // Giá trị trước đó để tính biến thiên
float prev_h_dht = 0;

// Biến lưu trữ lags (cho dữ liệu lag3 nếu cần)
float temp_dht_lag3 = 0;
float hum_dht_lag3 = 0;

// Biến trạng thái
unsigned long lastSendTime = 0;
unsigned long lastPredictTime = 0;
unsigned long lastBlinkTime = 0;
unsigned long lastApiTime = 0;
unsigned long lastReadTime = 0;

const unsigned long READ_INTERVAL = 10000;     // 10 giây
const unsigned long API_INTERVAL = 300000;     // 5 phút
const unsigned long SEND_INTERVAL = 60000;     // 1 phút
const unsigned long PREDICT_INTERVAL = 60000;  // 1 phút

bool ledState = false;
bool isPredictedAlert = false;

// Biến đếm dữ liệu đã ghi
int dataCount = 0;

// Buffer để lưu trữ các giá trị trước đó làm lag
#define MAX_HISTORY 5
float temp_history[MAX_HISTORY] = {0};
float hum_history[MAX_HISTORY] = {0};
int history_index = 0;

// Khởi tạo DHT
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  // Khởi tạo Serial
  Serial.begin(115200);
  delay(1000);  // Đợi Serial ổn định
  
  // Hiển thị thông tin khởi động
  Serial.println("\n=================================================");
  Serial.println("  Hệ Thống Giám Sát và Dự Báo Môi Trường Thông Minh");
  Serial.println("=================================================");
  
  // Khởi tạo chân GPIO
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);  // Tắt LED lúc khởi động
  
  // Khởi tạo cảm biến
  dht.begin();
  Serial.println("✓ Đã khởi tạo cảm biến DHT11");
  
  // Kết nối WiFi
  connectWiFi();
  
  // Cấu hình thời gian
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  Serial.println("✓ Đã cấu hình NTP");
  
  // Kiểm tra thời gian
  struct tm timeinfo;
  if(getLocalTime(&timeinfo)){
    Serial.print("Thời gian hiện tại: ");
    Serial.println(&timeinfo, "%A, %d %B %Y %H:%M:%S");
  } else {
    Serial.println("⚠️ Chưa đồng bộ được thời gian");
  }
  
  // Hiển thị thông tin về mô hình AI
  Serial.println("\n=== THÔNG TIN MÔ HÌNH AI ===");
  Serial.printf("Số lượng đặc trưng: %d\n", NUM_FEATURES);
  Serial.println("Hệ số mô hình nhiệt độ:");
  Serial.printf("- Intercept: %.4f\n", temp_intercept);
  Serial.println("Hệ số mô hình độ ẩm:");
  Serial.printf("- Intercept: %.4f\n", hum_intercept);
  
  // Hiển thị ngưỡng cảnh báo
  Serial.println("\n=== NGƯỠNG CẢNH BÁO ===");
  Serial.printf("Nhiệt độ: %.1f°C\n", TEMP_THRESHOLD);
  Serial.printf("Độ ẩm: %.1f%%\n", HUM_THRESHOLD);
  
  // Nhấp nháy LED để thông báo khởi động thành công
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }
  
  Serial.println("\n✓ Khởi động hệ thống hoàn tất");
  Serial.println("=================================================");
}

void loop() {
  // Kiểm tra kết nối WiFi và kết nối lại nếu mất kết nối
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("📡 Mất kết nối WiFi, đang kết nối lại...");
    connectWiFi();
  }
  
  // Đọc dữ liệu từ cảm biến theo chu kỳ
  unsigned long currentMillis = millis();
  if (currentMillis - lastReadTime >= READ_INTERVAL) {
    readSensorData();
    lastReadTime = currentMillis;
  }
  
  // Lấy dữ liệu API theo chu kỳ
  if (currentMillis - lastApiTime >= API_INTERVAL) {
    getWeatherData();
    lastApiTime = currentMillis;
  }
  
  // Dự đoán theo chu kỳ
  if (currentMillis - lastPredictTime >= PREDICT_INTERVAL && dht_ready() && api_ready()) {
    predictWeather();
    lastPredictTime = currentMillis;
  }
  
  // Gửi dữ liệu lên Sheets theo chu kỳ
  if (currentMillis - lastSendTime >= SEND_INTERVAL && dht_ready() && api_ready()) {
    sendToSheets(last_t_dht, last_h_dht, last_t_api, last_h_api, predicted_temp, predicted_hum);
    lastSendTime = currentMillis;
  }
  
  // Xử lý đèn LED
  handleLED();
  
  // Đợi một khoảng thời gian ngắn trước khi lặp lại
  delay(100);
}

// Kiểm tra xem dữ liệu DHT đã sẵn sàng chưa
bool dht_ready() {
  return !isnan(last_t_dht) && !isnan(last_h_dht);
}

// Kiểm tra xem dữ liệu API đã sẵn sàng chưa
bool api_ready() {
  return !isnan(last_t_api) && !isnan(last_h_api);
}

// Cập nhật lịch sử dữ liệu
void updateDataHistory(float temp, float hum) {
  // Cập nhật vị trí trong mảng lịch sử
  history_index = (history_index + 1) % MAX_HISTORY;
  
  // Lưu giá trị hiện tại vào lịch sử
  temp_history[history_index] = temp;
  hum_history[history_index] = hum;
  
  // Cập nhật giá trị lag3 (3 mẫu trước)
  int lag3_index = (history_index + MAX_HISTORY - 3) % MAX_HISTORY;
  temp_dht_lag3 = temp_history[lag3_index];
  hum_dht_lag3 = hum_history[lag3_index];
  
  // Nếu giá trị là 0 (chưa có dữ liệu), sử dụng giá trị hiện tại
  if (temp_dht_lag3 == 0) temp_dht_lag3 = temp;
  if (hum_dht_lag3 == 0) hum_dht_lag3 = hum;
}

// Đọc dữ liệu từ cảm biến DHT11
void readSensorData() {
  // Lưu giá trị trước đó
  prev_t_dht = last_t_dht;
  prev_h_dht = last_h_dht;
  
  // Đọc giá trị mới
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  
  if (isnan(t) || isnan(h)) {
    Serial.println("❌ Không đọc được DHT11");
  } else {
    last_t_dht = t;
    last_h_dht = h;
    
    // Cập nhật lịch sử dữ liệu
    updateDataHistory(t, h);
    
    Serial.printf("📊 DHT11: %.1f°C, %.1f%%\n", t, h);
  }
}

// Kết nối WiFi
void connectWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("📡 Kết nối WiFi");
  
  unsigned long startAttemptTime = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < 10000) {
    Serial.print(".");
    delay(500);
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("✓ Đã kết nối WiFi!");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println();
    Serial.println("❌ Không kết nối được WiFi. Sẽ thử lại sau.");
  }
}

// Lấy dữ liệu từ OpenWeather API
void getWeatherData() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("❌ Không có kết nối WiFi để lấy dữ liệu API");
    return;
  }
  
  HTTPClient http;
  WiFiClient client;
  http.setTimeout(10000);  // 10 giây timeout
  
  String url = "http://api.openweathermap.org/data/2.5/weather?q=" + String(city) + "," + String(country) +
               "&appid=" + apiKey + "&units=" + units;
  
  http.begin(client, url);
  Serial.println("🌐 Đang lấy dữ liệu thời tiết từ API...");
  int httpCode = http.GET();
  
  if (httpCode == 200) {
    String payload = http.getString();
    StaticJsonDocument<1024> doc;
    DeserializationError error = deserializeJson(doc, payload);
    
    if (!error) {
      last_t_api = doc["main"]["temp"];
      last_h_api = doc["main"]["humidity"];
      Serial.printf("🌐 API: %.1f°C, %.1f%%\n", last_t_api, last_h_api);
      
      // Hiển thị thêm thông tin thời tiết nếu có
      if (doc.containsKey("weather") && doc["weather"].size() > 0) {
        const char* description = doc["weather"][0]["description"];
        Serial.printf("🌐 Điều kiện thời tiết: %s\n", description);
      }
    } else {
      Serial.println("❌ Lỗi giải mã JSON: " + String(error.c_str()));
    }
  } else {
    Serial.printf("❌ Lỗi API: %d\n", httpCode);
  }
  
  http.end();
}

// Gửi dữ liệu lên Google Sheets
bool sendToSheets(float t_dht, float h_dht, float t_api, float h_api, float t_pred, float h_pred) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("❌ Không có kết nối WiFi để gửi dữ liệu");
    return false;
  }
  
  // Tăng bộ đếm dữ liệu
  dataCount++;
  
  WiFiClientSecure client;
  client.setInsecure();  // Bỏ qua xác minh chứng chỉ SSL
  
  HTTPClient http;
  http.setTimeout(15000);  // 15 giây timeout
  
  // Tạo URL với dữ liệu
  String url = String(scriptUrl);
  url += "?temp_dht=" + String(t_dht, 2);
  url += "&hum_dht=" + String(h_dht, 2);
  url += "&temp_api=" + String(t_api, 2);
  url += "&hum_api=" + String(h_api, 2);
  url += "&temp_pred=" + String(t_pred, 2);
  url += "&hum_pred=" + String(h_pred, 2);
  url += "&status=" + String(isPredictedAlert ? "WARNING" : "NORMAL");
  url += "&nocache=" + String(random(100000));  // Ngăn cache
  
  http.begin(client, url);
  Serial.printf("📤 Gửi dữ liệu lên Google Sheets (#%d)...\n", dataCount);
  
  int httpCode = http.GET();
  bool success = false;
  
  if (httpCode > 0) {
    String payload = http.getString();
    Serial.printf("✓ Đã gửi dữ liệu, HTTP code: %d\n", httpCode);
    if (httpCode == 200) {
      Serial.println("✓ Phản hồi: " + payload);
      success = true;
    }
  } else {
    Serial.printf("❌ Gửi Google Sheets lỗi: %d\n", httpCode);
    Serial.println("❌ Lỗi: " + http.errorToString(httpCode));
  }
  
  http.end();
  return success;
}

// Dự đoán nhiệt độ và độ ẩm dựa trên mô hình AI với chuẩn hóa dữ liệu
void predictWeather() {
  // Kiểm tra dữ liệu đầu vào
  if (isnan(last_t_dht) || isnan(last_h_dht) || isnan(last_t_api) || isnan(last_h_api) ||
      isnan(prev_t_dht) || isnan(prev_h_dht)) {
    Serial.println("❌ Dữ liệu đầu vào không hợp lệ cho dự đoán");
    return;
  }
  
  // Lấy thời gian hiện tại
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    Serial.println("❌ Không lấy được thời gian");
    return;
  }
  
  // Chuẩn bị các đặc trưng gốc
  float raw_features[NUM_FEATURES];
  
  // Gán giá trị theo thứ tự đặc trưng như trong file model_coef.h
  // Đảm bảo thứ tự phải khớp với mô hình đã huấn luyện!
  
  // Xác định thứ tự đặc trưng dựa trên comment trong model_coef.h
  // temp_api, hum_api, temp_dht_lag1, temp_dht_lag3, hum_dht_lag1, hum_dht_lag3, 
  // temp_dht_diff, hum_dht_diff, temp_diff_dht_api, hum_diff_dht_api
  raw_features[0] = last_t_api;                   // temp_api
  raw_features[1] = last_h_api;                   // hum_api
  raw_features[2] = prev_t_dht;                   // temp_dht_lag1
  raw_features[3] = temp_dht_lag3;                // temp_dht_lag3
  raw_features[4] = prev_h_dht;                   // hum_dht_lag1
  raw_features[5] = hum_dht_lag3;                 // hum_dht_lag3
  raw_features[6] = last_t_dht - prev_t_dht;      // temp_dht_diff
  raw_features[7] = last_h_dht - prev_h_dht;      // hum_dht_diff
  raw_features[8] = last_t_dht - last_t_api;      // temp_diff_dht_api
  raw_features[9] = last_h_dht - last_h_api;      // hum_diff_dht_api
  
  // Hiển thị các đặc trưng gốc
  Serial.println("\n🔢 Đặc trưng gốc:");
  for (int i = 0; i < NUM_FEATURES; i++) {
    Serial.printf("  - Đặc trưng %d: %.2f\n", i, raw_features[i]);
  }
  
  // Chuẩn hóa các đặc trưng
  float features[NUM_FEATURES];
  for (int i = 0; i < NUM_FEATURES; i++) {
    features[i] = (raw_features[i] - feature_means[i]) / feature_scales[i];
  }
  
  // Hiển thị các đặc trưng đã chuẩn hóa
  Serial.println("🧮 Đặc trưng chuẩn hóa:");
  for (int i = 0; i < NUM_FEATURES; i++) {
    Serial.printf("  - Đặc trưng %d: %.2f\n", i, features[i]);
  }
  
  // Dự đoán nhiệt độ
  predicted_temp = temp_intercept;
  for (int i = 0; i < NUM_FEATURES; i++) {
    predicted_temp += temp_coef[i] * features[i];
  }
  
  // Dự đoán độ ẩm
  predicted_hum = hum_intercept;
  for (int i = 0; i < NUM_FEATURES; i++) {
    predicted_hum += hum_coef[i] * features[i];
  }
  
  // Kiểm tra giá trị dự đoán hợp lý
  if (isnan(predicted_temp) || predicted_temp < 0 || predicted_temp > 50) {
    Serial.println("⚠️ Dự đoán nhiệt độ không hợp lý, sử dụng dự đoán đơn giản");
    predicted_temp = 0.7 * last_t_api + 0.3 * last_t_dht;
  }
  
  if (isnan(predicted_hum) || predicted_hum < 0 || predicted_hum > 100) {
    Serial.println("⚠️ Dự đoán độ ẩm không hợp lý, sử dụng dự đoán đơn giản");
    predicted_hum = 0.7 * last_h_api + 0.3 * last_h_dht;
  }
  
  // Đảm bảo kết quả nằm trong phạm vi hợp lý
  predicted_temp = constrain(predicted_temp, 0, 50);
  predicted_hum = constrain(predicted_hum, 0, 100);
  
  // Hiển thị kết quả dự đoán chi tiết
  Serial.println("\n=== KẾT QUẢ DỰ ĐOÁN ===");
  Serial.printf("🔮 Dự đoán (1 giờ tới): %.1f°C, %.1f%%\n", predicted_temp, predicted_hum);
  
  // So sánh với giá trị hiện tại
  Serial.println("\n📊 So sánh với giá trị hiện tại:");
  Serial.printf("- Nhiệt độ hiện tại: %.1f°C -> Dự đoán: %.1f°C (thay đổi: %.1f°C)\n", 
                last_t_dht, predicted_temp, predicted_temp - last_t_dht);
  Serial.printf("- Độ ẩm hiện tại: %.1f%% -> Dự đoán: %.1f%% (thay đổi: %.1f%%)\n", 
                last_h_dht, predicted_hum, predicted_hum - last_h_dht);
  
  // So sánh với giá trị API
  Serial.println("\n🌐 So sánh với dữ liệu API:");
  Serial.printf("- Nhiệt độ API: %.1f°C -> Dự đoán: %.1f°C (chênh lệch: %.1f°C)\n", 
                last_t_api, predicted_temp, predicted_temp - last_t_api);
  Serial.printf("- Độ ẩm API: %.1f%% -> Dự đoán: %.1f%% (chênh lệch: %.1f%%)\n", 
                last_h_api, predicted_hum, predicted_hum - last_h_api);
  
  // So sánh với ngưỡng
  Serial.println("\n⚠️ So sánh với ngưỡng cảnh báo:");
  Serial.printf("- Nhiệt độ dự đoán: %.1f°C %s Ngưỡng: %.1f°C\n", 
                predicted_temp, 
                predicted_temp > TEMP_THRESHOLD ? ">" : "<=", 
                TEMP_THRESHOLD);
  Serial.printf("- Độ ẩm dự đoán: %.1f%% %s Ngưỡng: %.1f%%\n", 
                predicted_hum, 
                predicted_hum > HUM_THRESHOLD ? ">" : "<=", 
                HUM_THRESHOLD);
  
  // Kiểm tra cảnh báo dự đoán
  bool wasPredictedAlert = isPredictedAlert;
  isPredictedAlert = (predicted_temp > TEMP_THRESHOLD || predicted_hum > HUM_THRESHOLD);
  
  if (isPredictedAlert && !wasPredictedAlert) {
    Serial.println("\n⚠️ CẢNH BÁO SỚM: Dự báo sẽ vượt ngưỡng trong 1 giờ tới!");
    if (predicted_temp > TEMP_THRESHOLD) {
      Serial.printf("🔥 Nhiệt độ dự đoán cao: %.1f°C > %.1f°C\n", predicted_temp, TEMP_THRESHOLD);
    }
    if (predicted_hum > HUM_THRESHOLD) {
      Serial.printf("💧 Độ ẩm dự đoán cao: %.1f%% > %.1f%%\n", predicted_hum, HUM_THRESHOLD);
    }
  } else if (!isPredictedAlert && wasPredictedAlert) {
    Serial.println("\n✓ Dự báo trở về mức bình thường");
  } else if (isPredictedAlert) {
    Serial.println("\n⚠️ Vẫn trong tình trạng cảnh báo dự đoán!");
  } else {
    Serial.println("\n✓ Tình trạng bình thường");
  }
}

// Xử lý đèn LED dựa trên điều kiện dự đoán
void handleLED() {
  unsigned long currentMillis = millis();
  
  if (isPredictedAlert) {
    // Nhấp nháy nếu dự đoán vượt ngưỡng
    if (currentMillis - lastBlinkTime >= 500) {
      lastBlinkTime = currentMillis;
      ledState = !ledState;
      digitalWrite(LED_PIN, ledState ? HIGH : LOW);
    }
  } else {
    // Tắt LED nếu mọi thứ bình thường
    digitalWrite(LED_PIN, LOW);
    ledState = false;
  }
}