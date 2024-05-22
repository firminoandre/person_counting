import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading
import time

# Configurações do YOLO e áreas de interesse
model = YOLO('yolov8s.pt')
area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]
area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]

app = Flask(__name__)
socketio = SocketIO(app)

person_exiting = set()
person_entering = set()

def process_video():
    cap = cv2.VideoCapture('person3.mp4')
    class_list = open("coco.txt").read().strip().split("\n")
    tracker = Tracker()
    person_entering_ids = {}
    person_exiting_ids = {}
    global person_exiting, person_entering

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 2 != 0:
            continue
        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        list = []

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if 'person' in c:
                list.append([x1, y1, x2, y2])
        b_box_id = tracker.update(list)

        for b_box in b_box_id:
            x3, y3, x4, y4, id = b_box

            results_in_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
            if results_in_area2 >= 0:
                person_entering_ids[id] = (x4, y4)
            if id in person_entering_ids:
                results_in_area1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
                if results_in_area1 >= 0:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
                    person_entering.add(id)

            results_in_area1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
            if results_in_area1 >= 0:
                person_exiting_ids[id] = (x4, y4)
            if id in person_exiting_ids:
                results_in_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
                if results_in_area2 >= 0:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
                    person_exiting.add(id)

        cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
        cv2.putText(frame, str('1'), (504, 471), cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 0), 1)

        cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
        cv2.putText(frame, str('2'), (466, 485), cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 0), 1)

        print('entrando', len(person_entering))
        print('saind', len(person_exiting))

        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

def send_person_exiting_count():
    while True:
        socketio.emit('update_count', {
            'exiting_count': len(person_exiting),
            'entering_count': len(person_entering)
        })
        time.sleep(1)

if __name__ == '__main__':
    video_thread = threading.Thread(target=process_video)
    video_thread.start()

    count_thread = threading.Thread(target=send_person_exiting_count)
    count_thread.start()

    socketio.run(app, allow_unsafe_werkzeug=True)
