# Real-Time Basketball Player Tracking and Gesture Recognition

This project is the core computer vision module for a basketball training robot. Using a single webcam, it tracks a player's position in 3D space, estimates their distance and angle, and recognizes a "raised hand" gesture to trigger actions like passing the ball.

---

## 🚀 Key Features

-   **Real-Time Player Detection**: Employs the YOLO model for fast and robust identification of the player in the frame.
-   **3D Position Estimation**: Calculates the player's real-world distance, horizontal angle (φ), and vertical angle (θ) relative to the camera.
-   **Gesture Recognition**: Detects when the player raises a hand above their head, providing a clear signal for the robot to act.
-   **Smooth & Stable Tracking**: Implements a **Kalman Filter** to smooth the tracking data, reducing jitter and providing stable output crucial for precise robotic control.

---

## 🛠️ How It Works: The Vision Pipeline

The system processes each camera frame through a multi-stage pipeline to achieve its goal:

1.  **Person Detection (YOLO)**: The `ultralytics YOLO` model scans the frame to detect any person and draws a bounding box around them. This creates a focused Region of Interest (ROI) for the next, more computationally intensive steps.

2.  **Pose Estimation (Mediapipe)**: Within the ROI, Google's `Mediapipe Pose` model is applied. It extracts a 33-point skeleton of the player, providing the precise pixel coordinates of key body joints like the nose, ankles, and wrists.

3.  **Calculation & Logic**:
    -   **Distance**: The distance to the player is estimated using the pinhole camera model, which relates the player's known real-world height to their measured height in pixels (from nose to ankles).
    -   **Angles**: The horizontal and vertical angles are calculated based on the deviation of the player's bounding box center from the image's center point.
    -   **Gesture Trigger**: The system activates the "raised hand" signal by checking if the player's wrist landmarks are vertically higher than their nose landmark.

4.  **State Smoothing (Kalman Filter)**:
    -   Raw position measurements from computer vision can be noisy and unstable. To solve this, a **Kalman Filter** is used to smooth the distance and angle values over time.
    -   The filter predicts the player's next state (position and velocity) and then corrects this prediction with the new measurement from the current frame. This results in a much more stable and reliable output, preventing erratic movements in a physical robot.

---

## 💻 Tech Stack

-   **Language**: Python
-   **Core Libraries**:
    -   `OpenCV` for camera interaction and image processing.
    -   `Ultralytics (YOLO)` for object detection.
    -   `Mediapipe` for pose estimation.
    -   `NumPy` for numerical operations.

---

## ⚙️ Setup and Installation

Follow these steps to run the project locally.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Imadlotfi1/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    A `requirements.txt` file should be included for easy installation.
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ How to Run

Once the setup is complete, ensure your webcam is connected and run the main script:

```bash
python main.py

