import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup
detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=".venv\Lib\site-packages\pose_landmarker_heavy.task"),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.2,    # Lower threshold
        min_pose_presence_confidence=0.1      # Very low
    )
)

# Skeleton (prioritize body over face)
MAIN_SKELETON = [
    (11, 12), (11, 23), (12, 24), (23, 24),  # Torso
    (11, 13), (13, 15),                      # Left arm
    (12, 14), (14, 16),                      # Right arm
    (23, 25), (25, 27),                      # Left leg
    (24, 26), (26, 28)                       # Right leg
]

FACE_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8)
]
height, width = 480, 640
blank = np.zeros((height, width, 3), dtype=np.uint8)
cap = cv2.VideoCapture(0)
cv2.namedWindow('hasil', cv2.WINDOW_NORMAL)
cv2.resizeWindow('hasil', 640, 480)

last_pose = None
skip_counter = 0
status = ""
status2 = ""

print("Face can be hidden - Press 'q' to quit")



# def safe_imshow(window_name, image):
#     """Safe version of cv2.imshow with None check"""
#     if image is None:
#         print(f"⚠️ Warning: Cannot display {window_name} - image is None")
#         # Create a placeholder black image
#         placeholder = np.zeros((300, 400, 3), dtype=np.uint8)
#         cv2.putText(placeholder, "NO IMAGE", (100, 150), 
#         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         cv2.imshow(window_name, placeholder)
#     else:
#         cv2.imshow(window_name, image)

def is_hands_up_relative(landmarks):
    """
    Cek tinggi tangan relatif terhadap kepala
    """
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    nose = landmarks[0]  # Hidung
    
    # Tangan harus di atas/di level kepala
    left_up = left_wrist.y < nose.y + 0.1
    right_up = right_wrist.y < nose.y + 0.1
    
    return right_up and not left_up

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Detect
    result = detector.detect(
        mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
    )
    
    current_pose = None
    
    if result.pose_landmarks:
        current_pose = result.pose_landmarks[0]
        last_pose = current_pose
        skip_counter = 0
        if is_hands_up_relative(current_pose):
            # print("angkat tangan")
            status2 = "angkat tangan kiri"
            # img = cv2.imread('gambars/angkatTanganKiri1.jpg')  # Atau buat gambar
            # cv2.imshow('Window 1', img)
            # print('gambars\angkatTanganKiri1.jpg')
            # safe_imshow('hasil', img)
        else:
            status2 = "ga ngapa ngapain"
            # safe_imshow('hasil', blank)
    elif last_pose and skip_counter < 10:  # Keep last pose for 10 frames
        current_pose = last_pose
        skip_counter += 1
    else:
        last_pose = None
    
    # Draw if we have a pose
    if current_pose:
        h, w = frame.shape[:2]
        
        # Draw MAIN skeleton (body) - ALWAYS
        for start, end in MAIN_SKELETON:
            if start < len(current_pose) and end < len(current_pose):
                x1 = int(current_pose[start].x * w)
                y1 = int(current_pose[start].y * h)
                x2 = int(current_pose[end].x * w)
                y2 = int(current_pose[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw FACE skeleton only if nose is visible
        if (0 < len(current_pose) and 
            hasattr(current_pose[0], 'visibility') and 
            current_pose[0].visibility > 0.1):
            
            for start, end in FACE_SKELETON:
                if start < len(current_pose) and end < len(current_pose):
                    x1 = int(current_pose[start].x * w)
                    y1 = int(current_pose[start].y * h)
                    x2 = int(current_pose[end].x * w)
                    y2 = int(current_pose[end].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
        
        # Draw key points
        key_indices = [11, 12, 23, 24, 13, 14, 15, 16, 25, 26, 27, 28]
        for idx in key_indices:
            if idx < len(current_pose):
                x = int(current_pose[idx].x * w)
                y = int(current_pose[idx].y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        status = "rkhaf"
        if skip_counter > 0:
            status += f" (cached: {skip_counter})"
    
    else:
        status = "❌ No pose"
        status2=""
        # safe_imshow('hasil', blank)
    
    # Display status
    cv2.putText(frame, status, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 0) if current_pose else (0, 0, 255), 2)
    
    
    cv2.putText(frame, status2, (20, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 0) if current_pose else (0, 0, 255), 2)
    cv2.imshow('Pose - Body stays when face hidden', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()