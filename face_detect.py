from deepface import DeepFace
import cv2
import csv
import time


reference_img_path = "proctoring/src/MY_PIC.jpg"
reference_img = cv2.imread(reference_img_path)
if reference_img is None:
    raise Exception(f"Image not found at {reference_img_path}")


ans = []
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Failed to open webcam")

interval = 1  
last_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if current_time - last_time > interval:
            last_time = current_time
            try:
                result = DeepFace.verify(reference_img, frame, model_name='VGG-Face', detector_backend='opencv')
                
                if result["verified"]:
    
                    ans.append(0)
                else:
                    print("not detected")
                    ans.append(0.95)

            except Exception as e:
                print("not detected")
                ans.append(0.75)

        cv2.imshow('Face Verification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

    # Write results to CSV
    with open('verification_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for value in ans:
            writer.writerow([value])
