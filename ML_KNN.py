import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset dari file CSV
color_data = pd.read_csv('colors.csv')
X = color_data[['B', 'G', 'R']].values
y = color_data['color_name'].values

# Normalisasi Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Prediksi data test
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Simpan model dan scaler
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(scaler, 'knn_scaler.pkl')

# Muat model KNN dan scaler
knn = joblib.load('knn_model.pkl')
scaler = joblib.load('knn_scaler.pkl')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Ambil pixel tengah gambar
    height, width, _ = frame.shape
    pixel_center = frame[height//2, width//2]
    
    # Normalisasi pixel sebelum prediksi
    pixel_center_scaled = scaler.transform([pixel_center])
    
    # Prediksi warna
    color_pred = knn.predict(pixel_center_scaled)[0]
    
    # Tampilkan warna pada frame
    cv2.putText(frame, f'Color: {color_pred}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()