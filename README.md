# Vehicle-Speed-Measurement-using-YOLOv8-and-Object-Tracking

## 📌 Project Overview
This project demonstrates how to measure vehicle speed from video footage using **YOLOv8 for object detection** and **tracking algorithms**. The key steps include:
- Detecting vehicles in each frame using **YOLOv8**.
- Tracking vehicles across consecutive frames using **ByteTrack**.
- Calculating speed by measuring pixel displacement and converting it to real-world speed using a **scaling factor**.

---

## 📂 Project Workflow

### 1️⃣ Object Detection & Tracking
- YOLOv8 detects cars in each frame.
- ByteTrack assigns a unique ID to track each car across multiple frames.

### 2️⃣ Scaling Factor Calculation
To convert **pixel movement** into **real-world speed**, we calculate the **scaling factor** as:

```
Scaling Factor = Real-world width (m) / Measured pixel width
```

For example, if a car of **1.7m width** appears **85 pixels** wide in the image:

```
Scaling Factor = 1.7 / 85 = 0.02 m/pixel
```

### 3️⃣ Speed Calculation: Example
Using a **Frame Rate (FPS) of 25**, consider two consecutive frames:

- **Frame A**: Bounding box center at **(100, 200)**
- **Frame B**: Bounding box center at **(106, 200)**

#### Step 1: Compute Pixel Distance
```
Pixel Distance = sqrt((106 - 100)² + (200 - 200)²)
               = sqrt(6² + 0²)
               = sqrt(36)
               = 6 pixels
```

#### Step 2: Convert to Real-World Distance
```
Real-World Distance = Pixel Distance × Scaling Factor
                    = 6 × 0.02
                    = 0.12 meters
```

#### Step 3: Calculate Speed
```
Time per Frame = 1 / FPS = 1 / 25 = 0.04 seconds
Speed (m/s) = Real-World Distance / Time per Frame
            = 0.12 / 0.04
            = 3 m/s
Speed (km/h) = Speed (m/s) × 3.6
             = 3 × 3.6
             = 10.8 km/h
```

### 4️⃣ Speed Smoothing
- A **moving average window** is applied over multiple frames to smooth out fluctuations in speed readings.

### 5️⃣ Real-Time Visualization
- Bounding boxes, **track IDs**, and **speed values** are displayed on the video.

---

## 🚀 Technologies Used
- **Python** 🐍
- **OpenCV** for image processing
- **YOLOv8** for object detection
- **ByteTrack** for multi-object tracking
- **NumPy** for mathematical operations

---

## 📌 How to Run the Project
1. Install dependencies:
   ```bash
   pip install ultralytics opencv-python numpy
   ```
2. Run the speed measurement script:
   ```bash
   python speed_measurement.py
   ```
3. Adjust the **scaling factor** as needed for different camera perspectives.

---

## 📌 Notes
- **Scaling Factor** must be calibrated using a real-world object for accurate speed measurement.
- **Higher confidence thresholds** (e.g., 0.5) can reduce false positives in detection.

---

## 📩 Get in Touch!
If you're interested in learning more about this project, feel free to reach out! 🚗📷⚡
tkseneee@gmail.com

#ComputerVision #YOLOv8 #ObjectDetection #Tracking #SpeedMeasurement #MachineLearning #AI #Python #OpenCV #DeepLearning
