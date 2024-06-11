import cv2
import dlib
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Load employees database
def load_employees_db():
    df = pd.read_csv('employees.csv')
    print("Loaded employee data:")
    print(df)
    df['encodings'] = df['photo_path'].apply(lambda x: get_face_encodings(cv2.imread(x)))
    return df

# Extract face encodings from an image
def get_face_encodings(image):
    if image is None:
        print("Error loading image.")
        return []

    faces = detector(image)
    encodings = []
    for face in faces:
        shape = predictor(image, face)
        encoding = np.array(face_rec_model.compute_face_descriptor(image, shape))
        encodings.append(encoding)
    if not encodings:
        print("No faces found in the image.")
    return encodings

# Function to process a frame and identify faces
def process_frame(frame, employees_db, identified_names):
    encodings = get_face_encodings(frame)
    current_time = time.time()

    for encoding in encodings:
        for index, row in employees_db.iterrows():
            target_name = row['name']
            target_encodings = row['encodings']
            if not target_encodings:
                print(f"No encodings for {target_name}.")
                continue
            for target_encoding in target_encodings:
                dist = np.linalg.norm(encoding - target_encoding)
                if dist < 0.65:  # Slightly relaxed threshold
                    identified_names[target_name] = current_time

    return frame

# Function to search for faces on camera
def search_on_camera():
    employees_db = load_employees_db()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)  # Lower resolution
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0
    process_every_n_frames = 5  # Process every 5th frame
    identified_names = {}  # Dictionary to store identified names and timestamps
    executor = ThreadPoolExecutor(max_workers=1)
    display_duration = 2  # Display duration in seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % process_every_n_frames == 0:
            future = executor.submit(process_frame, frame.copy(), employees_db, identified_names)
            future.result()

        # Display the names that have been identified within the last `display_duration` seconds
        current_time = time.time()
        for name, timestamp in identified_names.items():
            if current_time - timestamp <= display_duration:
                cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    try:
        search_on_camera()
    except KeyboardInterrupt:
        pass

# Run the script
if __name__ == "__main__":
    main()

