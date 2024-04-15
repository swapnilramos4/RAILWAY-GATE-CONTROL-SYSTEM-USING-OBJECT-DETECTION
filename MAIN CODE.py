import random
import cv2
from time import sleep
from pyfirmata import Arduino, SERVO, OUTPUT
import threading
from ultralytics import YOLO
import torch

port = 'COM6'  # Arduino Port Pin
pin = 10  # Signal Pin for Servo
buzzer_pin = 11  # Signal Pin for Buzzer
board = Arduino(port)
board.digital[pin].mode = SERVO
board.digital[buzzer_pin].mode = OUTPUT  # Set buzzer pin as output

# Open the class list file
with open("utils/coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Generate random colors for class list
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(class_list))]

# Load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Capture video from webcam
url='http://192.168.29.40:8080/video'
vid = cv2.VideoCapture(0)

def rotateservo(pin, angle):
    board.digital[pin].write(angle)

def buzzer_control(state):
    board.digital[buzzer_pin].write(state)

def servo_control(stop_event):
    servo_position = 0  # Initial servo position
    while not stop_event.is_set():
        if is_person_detected and servo_position != 90:
            buzzer_control(1)  # Turn on the buzzer
            for i in range(servo_position, 90):
                rotateservo(pin, i)
                sleep(0.005)  # Adjust the delay to control speed (faster)
                servo_position = i
        elif not is_person_detected and servo_position != 0:
            buzzer_control(0)  # Turn off the buzzer
            for i in range(servo_position, 0, -1):
                rotateservo(pin, i)
                sleep(0.005)  # Adjust the delay to control speed (faster)
                servo_position = i
        sleep(0.1)  # Additional delay after each complete movement

is_person_detected = False

def object_detection(stop_event):
    global is_person_detected
    while not stop_event.is_set():
        ret, frame = vid.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        detect_params = model.predict(source=[frame], conf=0.5, save=False)

        person_detected = False
        for box in detect_params[0].boxes:
            clsID_cpu = box.cls.cpu()
            clsID = clsID_cpu.numpy()[0]
            if clsID == 2:  # Check if detected class is person
                person_detected = True
                xyxy = box.xyxy.cpu().numpy()
                pt1 = (int(xyxy[0][0]), int(xyxy[0][1]))
                pt2 = (int(xyxy[0][2]), int(xyxy[0][3]))
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, class_list[int(clsID)], pt1, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                break

        is_person_detected = person_detected

        cv2.imshow("ObjectDetection", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            stop_event.set()  # Set the event to stop the threads
            break

# Create stop events for the threads
stop_event_servo = threading.Event()
stop_event_detection = threading.Event()

# Start the threads
thread_servo = threading.Thread(target=servo_control, args=(stop_event_servo,))
thread_detection = threading.Thread(target=object_detection, args=(stop_event_detection,))
thread_servo.start()
thread_detection.start()

# Wait for the threads to finish or until 'q' is pressed
while True:
    if not thread_servo.is_alive() or not thread_detection.is_alive():
        break
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Stop event set by user.")
        stop_event_servo.set()
        stop_event_detection.set()
        break

# Release video capture
vid.release()
cv2.destroyAllWindows()