from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# Initialize the pose estimation model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the overlay image
overlay = cv2.imread("shirt.png", cv2.IMREAD_UNCHANGED)

# Define the keypoints for the shoulders and the torso
shoulder_keypoints = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]
torso_keypoints = [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to RGB format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with the pose estimation model
        results = pose.process(image)

        # Check if the pose was detected
        if results.pose_landmarks:
            # Get the keypoints for the shoulders and the torso
            shoulder_points = []
            torso_points = []
            for keypoint in shoulder_keypoints:
                landmark = results.pose_landmarks.landmark[keypoint]
                shoulder_points.append((landmark.x * frame.shape[1], landmark.y * frame.shape[0]))
            for keypoint in torso_keypoints:
                landmark = results.pose_landmarks.landmark[keypoint]
                torso_points.append((landmark.x * frame.shape[1], landmark.y * frame.shape[0]))

            # Calculate the width and height of the overlay based on the keypoints
            width = int(np.linalg.norm(np.array(shoulder_points[0]) - np.array(shoulder_points[1])) * 2)
            height = int(np.linalg.norm(np.array(torso_points[0]) - np.array(torso_points[1])) * 2)

            # Ensure the dimensions are positive and non-zero
            width = max(1, width)
            height = max(1, height)

            # Calculate the transformation matrix to warp the overlay
            src_points = np.array(shoulder_points + torso_points, dtype=np.float32)
            dst_points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_points, dst_points)

            # Warp the overlay to fit the person's body shape
            overlay_resized = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_AREA)
            overlay_warped = cv2.warpPerspective(overlay_resized, M, (frame.shape[1], frame.shape[0]))

            # Create a mask from the alpha channel of the overlay
            mask = overlay_warped[:, :, 3] / 255.0
            inverse_mask = 1.0 - mask

            # Blend the overlay with the original frame
            for c in range(3):
                frame[:, :, c] = (inverse_mask * frame[:, :, c] + mask * overlay_warped[:, :, c])

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in the format required by Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
