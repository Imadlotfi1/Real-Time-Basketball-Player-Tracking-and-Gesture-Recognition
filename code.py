import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import math

# --- Main Settings ---
# All the main parameters are here so they're easy to change.
CAMERA_INDEX = 0
PLAYER_HEIGHT_METERS = 1.70  # Needed for the distance calculation.
CAMERA_RESOLUTION = (1280, 720)
YOLO_MODEL_PATH = "yolov8n.pt" # Using yolov8n because it's fast and good enough for detecting one person.

class PlayerTracker:
    """
    This class bundles everything together for tracking the player.
    Decided to use a class to avoid messy global variables.
    """
    def __init__(self):
        # This is the setup method. It runs once when we create the tracker.
        print("Loading models, this might take a moment...")
        
        # Load the main models we'll need.
        self.model = YOLO(YOLO_MODEL_PATH)
        self.pose_detector = mp.solutions.pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_pose = mp.solutions.pose
        print("Models loaded successfully.")

        # Camera properties needed for calculations.
        self.focal_length = self._estimate_focal_length()
        self.img_center = (CAMERA_RESOLUTION[0] // 2, CAMERA_RESOLUTION[1] // 2)

        # The Kalman filter is the key for smooth tracking.
        # Raw measurements from the camera are way too jittery for a robot.
        self.kalman = self._setup_kalman_filter()

        # This will store the last known good position.
        # When the player raises their hand, we'll use this stable position to aim,
        # not the potentially blurry one from the current frame.
        self.last_known_position = {
            "distance": 0.0, "angle_phi": 0.0, "angle_theta": 0.0,
            "is_right": False, "hand_raised": False
        }
        self.hand_is_up = False

    def _estimate_focal_length(self, fov_degrees=60):
        # This is a decent estimate for a standard webcam's focal length.
        # For better accuracy, I should write a proper camera calibration script later.
        return (CAMERA_RESOLUTION[0] / 2) / math.tan(math.radians(fov_degrees / 2))

    def _setup_kalman_filter(self):
        # Setting up the matrices for the Kalman filter.
        # These values control how much it smooths vs. how fast it reacts to new movements.
        state_size = 6  # [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        meas_size = 3   # [pos_x, pos_y, pos_z]
        kalman = cv2.KalmanFilter(state_size, meas_size)
        
        dt = 1.0
        kalman.transitionMatrix = np.array([
            [1,0,0,dt,0,0], [0,1,0,0,dt,0], [0,0,1,0,0,dt],
            [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]
        ], dtype=np.float32)

        kalman.measurementMatrix = np.eye(meas_size, state_size, dtype=np.float32)
        # These noise values are found by experimenting.
        # A higher measurementNoise means we trust the camera less (more smoothing).
        kalman.processNoiseCov = np.eye(state_size, dtype=np.float32) * 1e-4
        kalman.measurementNoiseCov = np.eye(meas_size, dtype=np.float32) * 1e-2
        return kalman

    def process_frame(self, frame):
        # This is the main processing pipeline for each camera frame.
        # 1. Find the person with YOLO.
        results = self.model(frame, verbose=False, classes=[0])
        
        if results and results[0].boxes:
            box = results[0].boxes.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, box)

            # 2. Cut out the person (Region of Interest) and analyze their pose.
            roi = frame[y1:y2, x1:x2]
            current_data = self._analyze_pose_in_roi(roi, (x1, y1, x2, y2))
            
            if current_data is None:
                self.hand_is_up = False
                return frame

            # 3. Decide if it's a gesture or a normal movement.
            if current_data["hand_raised"]:
                if not self.hand_is_up:
                    # The moment the hand goes up, we trigger the action.
                    display_data = self.last_known_position.copy()
                    display_data["hand_raised"] = True
                    print(f"ACTION: Hand raised! Aiming at: {display_data}")
                    self.hand_is_up = True
                else:
                    # Hand is still up, keep showing the same stable position.
                    display_data = self.last_known_position
                    display_data["hand_raised"] = True
            else:
                # Hand is down, so we do normal tracking.
                self.hand_is_up = False
                # 4. Smooth the measurements with the Kalman filter.
                display_data = self._filter_measurements(current_data)
                # And save this as the new last known good position.
                self.last_known_position.update(display_data)
            
            # 5. Draw the results on the screen.
            frame = self._draw_info(frame, (x1, y1, x2, y2), display_data)

        return frame

    def _analyze_pose_in_roi(self, roi, box_coords):
        # Check if the ROI is valid before processing.
        h, w, _ = roi.shape
        if h == 0 or w == 0: return None
            
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(rgb_roi)

        if not results.pose_landmarks: return None

        lm = results.pose_landmarks.landmark
        
        # Get key body landmarks to calculate height in pixels.
        head_y = lm[self.mp_pose.PoseLandmark.NOSE].y * h
        foot_y = max(lm[self.mp_pose.PoseLandmark.LEFT_ANKLE].y, lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y) * h
        pixel_height = abs(foot_y - head_y)

        # If the detected person is too small, the pose is usually wrong. So we ignore it.
        if pixel_height < 50: return None

        # This is the main gesture logic: if a wrist is higher than the nose, signal "hand up".
        left_wrist_y = lm[self.mp_pose.PoseLandmark.LEFT_WRIST].y * h
        right_wrist_y = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].y * h
        hand_raised = left_wrist_y < head_y or right_wrist_y < head_y
        
        if hand_raised:
            return {"hand_raised": True}

        # If hand is not raised, calculate the distance and angles.
        # The classic pinhole camera formula. It's surprisingly effective.
        distance = (self.focal_length * PLAYER_HEIGHT_METERS) / pixel_height
        
        box_center_x = ((box_coords[0] + box_coords[2]) / 2) - self.img_center[0]
        box_center_y = self.img_center[1] - ((box_coords[1] + box_coords[3]) / 2)

        angle_phi = math.degrees(math.atan2(box_center_x, self.focal_length))
        angle_theta = math.degrees(math.atan2(box_center_y, self.focal_length))

        return {
            "distance": distance, "angle_phi": angle_phi, "angle_theta": angle_theta,
            "is_right": box_center_x > 0, "hand_raised": False
        }

    def _filter_measurements(self, data):
        # Feed the new, noisy measurement to the filter...
        measurement = np.array([[data["distance"]], [data["angle_phi"]], [data["angle_theta"]]], dtype=np.float32)
        self.kalman.predict()
        # ...and get the corrected, smoother state back.
        corrected_state = self.kalman.correct(measurement)
        
        data["distance"] = float(corrected_state[0])
        data["angle_phi"] = float(corrected_state[1])
        data["angle_theta"] = float(corrected_state[2])
        return data

    def _draw_info(self, frame, box, data):
        # Draw all the info on the screen. Let's make it green for tracking, red for action.
        x1, y1, x2, y2 = box
        color = (0, 0, 255) if data.get('hand_raised') else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        dist_text = f"Distance: {data['distance']:.2f}m"
        angle_text = f"Angle: {data['angle_phi']:.1f} deg"
        hand_text = f"Hand Raised: {'YES!' if data.get('hand_raised') else 'No'}"
        
        # Putting text above the bounding box.
        cv2.putText(frame, dist_text, (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, angle_text, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, hand_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

def main():
    # This is the main entry point of the script.
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAMERA_INDEX}. Is it connected?")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    
    tracker = PlayerTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab a frame from the camera.")
            break
        
        # Flip the frame so it acts like a mirror. Much more intuitive for the user.
        frame = cv2.flip(frame, 1)

        processed_frame = tracker.process_frame(frame)

        cv2.imshow("Basketball Player Tracker", processed_frame)

        # Allow the user to press 'ESC' to quit the program.
        if cv2.waitKey(1) & 0xFF == 27:
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
