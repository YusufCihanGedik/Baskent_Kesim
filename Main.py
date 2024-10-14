import cv2
import numpy as np
from ultralytics import YOLO
import time
import csv
from datetime import datetime

model_path = r"/home/aisoft/Desktop/baskentkesim/best.pt"
video_path = r"/home/aisoft/Desktop/baskentkesim/videos/1.mp4"

original_points = np.array([(716, 746), (324, 1200), (1036, 1410), (1280, 812)])
m_home_point = np.array([(2290, 802), (2292, 894), (2378, 906), (2366, 796)])

scale_factor = 0.5
movement_threshold = 5  # Hareket kontrolü için eşik değeri

resized_home_point = np.array([(int(x * scale_factor), int(y * scale_factor)) for x, y in m_home_point])
resized_points = np.array([(int(x * scale_factor), int(y * scale_factor)) for x, y in original_points])

model = YOLO(model_path)

def writer_csv(file_name, data):
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def log_time_problem(issue_type, elapsed_time, frame_counter, current_time):
    writer_csv('time_issues.csv', [issue_type, elapsed_time, frame_counter, current_time])

def get_time_with_zero_milliseconds():
    now = datetime.now()
    now = now.replace(microsecond=0)
    return now  

# Hareket tespiti yapmak için 2 saniyede bir koordinatları kontrol eden fonksiyon
def process_videos(video_path, model, points, original_points, scale_factor, home_point, frame_skip_interval, interval_seconds):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Video açılmıyor"
    
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Videonun Gerçek FPS'i: {real_fps}")
    
    frame_duration = 1 / real_fps  
    prev_time = get_time_with_zero_milliseconds()  
    frame_counter = 0 
    total_process_time = 0 
    skipped_frames = 0
    
    global in_home  
    in_home = False
    
    # Zaman aralığında değişiklikleri kontrol etmek için
    prev_check_time = time.time()
    prev_px, prev_py = None, None  # 2 saniye önceki px, py değerlerini tutacak değişkenler

    while cap.isOpened():
        start_time = time.time()  
        
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # İşlenmesi gereken frame olup olmadığını kontrol et
        if frame_counter % frame_skip_interval != 0:
            continue

        img_height, img_width = frame.shape[:2]
        resized_img = cv2.resize(frame, (int(img_width * scale_factor), int(img_height * scale_factor)))

        results = model([resized_img], device="0", conf=0.20)

        for result in results:
            class_ids = result.boxes.cls.cpu().numpy()
            class_ids_list = class_ids.tolist()

            if len(result.keypoints.data) > 0:
                current_px, current_py = draw_keypoints(result.boxes, result.keypoints.data, resized_img, img_height, img_width, points, real_fps, class_ids_list, home_point)

                # Her 2 saniyede bir kontrol et
                current_time = time.time()
                if (current_time - prev_check_time) >= interval_seconds:
                    # 2 saniye önceki koordinatlar ile şimdiki koordinatları karşılaştır
                    if prev_px is not None and prev_py is not None:
                        diff_px = abs(current_px - prev_px)
                        diff_py = abs(current_py - prev_py)

                        # Zamanı float yerine datetime formatına dönüştürüyoruz
                        current_time_dt = datetime.fromtimestamp(current_time)

                        if diff_px > movement_threshold or diff_py > movement_threshold:
                            print(f"Hareket tespit edildi: px farkı = {diff_px}, py farkı = {diff_py}")
                            writer_csv('movement_detection.csv', [current_time_dt.strftime("%H:%M:%S"), "Hareket tespit edildi", diff_px, diff_py])
                        else:
                            print(f"Hareket tespit edilmedi: px farkı = {diff_px}, py farkı = {diff_py}")
                            writer_csv('movement_detection.csv', [current_time_dt.strftime("%H:%M:%S"), "Hareket tespit edilmedi", diff_px, diff_py])

                    # Şu anki koordinatları sakla ve zamanı güncelle
                    prev_px, prev_py = current_px, current_py
                    prev_check_time = current_time

        cv2.polylines(resized_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(resized_img, [home_point], isClosed=True, color=(0, 255, 0), thickness=2)

        current_time = get_time_with_zero_milliseconds()
        elapsed_time = (current_time - prev_time).total_seconds()  # Geçen süreyi saniye cinsinden hesapla

        if elapsed_time >= 1:  
            log_time_problem("Saniye Atlaması", elapsed_time, frame_counter, current_time.strftime("%H:%M:%S"))
            writer_csv('output.csv', [current_time.strftime("%H:%M:%S"), in_home])
            prev_time = current_time
        
        end_time = time.time()
        process_duration = end_time - start_time
        total_process_time += process_duration

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Kalan süre kadar bekle
        if process_duration < frame_duration:
            time.sleep(frame_duration - process_duration)

    cap.release()
    cv2.destroyAllWindows()

def draw_keypoints(boxes, keypoints, frame, img_height, img_width, points, real_fps, class_ids_list, home_point):
    global in_home  # global değişkenleri tanımlıyoruz
    in_home = False  # Varsayılan olarak False

    for i in range(len(boxes)):
        box = boxes.xywh[i].tolist()
        x_center, y_center, width, height = box
        x_min = int(max(0, (x_center - width / 2) * img_width))
        y_min = int(max(0, (y_center - height / 2) * img_height))
        x_max = int(min(img_width, (x_center + width / 2) * img_width))
        y_max = int(min(img_height, (y_center + height / 2) * img_height))

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        keypoints_person = keypoints[i].data.tolist()

        if class_ids_list[i] == 0:
            specific_keypoint = keypoints_person[0]
            px = int(min(max(0, specific_keypoint[0]), img_width))  
            py = int(min(max(0, specific_keypoint[1]), img_height))  
            visibility = specific_keypoint[2]

            if visibility > 0.5: 
                dist = cv2.pointPolygonTest(home_point, (px, py), False)
                if dist >= 0:
                    in_home = True

                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

                # Mevcut px ve py'yi return ile döndürüyoruz
                return px, py
        else:
            for kp in keypoints_person:
                px = int(min(max(0, kp[0]), img_width))
                py = int(min(max(0, kp[1]), img_height))
                visibility = kp[2]

                if visibility > 0.5:
                    cv2.circle(frame, (px, py), 5, (255, 0, 0), -1)

    # Eğer keypoint yoksa
    return None, None


# Video işleme fonksiyonunu 2 saniyelik aralıklarla çalıştır
process_videos(video_path, model, resized_points, original_points, scale_factor, resized_home_point, frame_skip_interval=1, interval_seconds=2)
