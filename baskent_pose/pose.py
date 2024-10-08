# import cv2
# import numpy as np
# from ultralytics import YOLO
# import csv
# import time

# model_path = r"C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\model\yolov8x-pose.pt"
# video_path = r"C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\video_img\kesim2.mp4"


# original_points = np.array([(716, 746), (324, 1200), (1036, 1410), (1280, 812)])

# scale_factor = 0.5  # %50 oranında küçültme

# with open('output.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Time", "In_ROI"])


# resized_points = np.array([(int(x * scale_factor), int(y * scale_factor)) for x, y in original_points])

# model = YOLO(model_path)

# def process_videos(video_path, model, points, original_points, scale_factor):
#     cap = cv2.VideoCapture(video_path)
#     assert cap.isOpened(), "Video açılmıyor"

#     real_fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"Videonun Gerçek FPS'i: {real_fps}")

#     elapsed_time = 0
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         img_height, img_width = frame.shape[:2]

#         resized_img = cv2.resize(frame, (int(img_width * scale_factor), int(img_height * scale_factor)))

#         results = model([resized_img], device="cuda", conf=0.30)

#         for result in results:
#             if len(result.keypoints.data) > 0:
#                 draw_keypoints(result.boxes, result.keypoints.data, resized_img, img_height, img_width, points, real_fps, elapsed_time)

#         cv2.polylines(resized_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

#         cv2.imshow('Frame', resized_img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):  
#             break

#         elapsed_time += 1 / real_fps

#     cap.release()
#     cv2.destroyAllWindows()

# pTime = 0 

# def draw_keypoints(boxes, keypoints, frame, img_height, img_width, points, real_fps, elapsed_time):
#     global pTime  
#     for i in range(len(boxes)):
#         box = boxes.xywh[i].tolist()
#         x_center, y_center, width, height = box
#         x_min = int(max(0, (x_center - width / 2) * img_width))
#         y_min = int(max(0, (y_center - height / 2) * img_height))
#         x_max = int(min(img_width, (x_center + width / 2) * img_width))
#         y_max = int(min(img_height, (y_center + height / 2) * img_height))

        
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#         keypoints_person = keypoints[i].data.tolist()

#         in_roi = False
#         for kp in keypoints_person:
#             px = int(min(max(0, kp[0]), img_width))
#             py = int(min(max(0, kp[1]), img_height))

#             visibility = kp[2]

#             if visibility < 0.5:
#                 visibility = 0 
#             else:
#                 visibility = 2
            
#             color = (0, 0, 255) if visibility == 0 else (0, 255, 0)
#             cv2.circle(frame, (px, py), 5, color, -1)  
#             if cv2.pointPolygonTest(points, (px, py), False) >= 0:
#                 in_roi = True
#                 break

#     with open('output.csv', mode='a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([elapsed_time, in_roi])

    
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime

#     cv2.putText(frame, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
#     # cv2.putText(frame, f'FPS: {int(real_fps)}', (70, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

#     print(in_roi)


# process_videos(video_path, model, resized_points, original_points, scale_factor)


import cv2
import numpy as np
from ultralytics import YOLO
import csv

model_path = r"C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\model\best.pt"
video_path = r"C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\video_img\kesim2.mp4"

# Orijinal çözünürlükteki poligon noktaları
original_points = np.array([(716, 746), (324, 1200), (1036, 1410), (1280, 812)])

scale_factor = 0.5  # %50 oranında küçültme

with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "In_ROI"])

# Yeniden boyutlandırılmış poligon noktalarını hesapla
resized_points = np.array([(int(x * scale_factor), int(y * scale_factor)) for x, y in original_points])

# Modeli başlat
model = YOLO(model_path)

def process_videos(video_path, model, points, original_points, scale_factor):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Video açılmıyor"

    real_fps = cap.get(cv2.CAP_PROP_FPS)  # Videonun FPS'ini al
    print(f"Videonun Gerçek FPS'i: {real_fps}")

    frame_time = 1 / real_fps  # Bir frame'in süresi
    frame_count = 0  # Toplam frame sayısı
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_height, img_width = frame.shape[:2]

        resized_img = cv2.resize(frame, (int(img_width * scale_factor), int(img_height * scale_factor)))

        results = model([resized_img], device="cuda", conf=0.30)

        for result in results:
            if len(result.keypoints.data) > 0:
                draw_keypoints(result.boxes, result.keypoints.data, resized_img, img_height, img_width, points, frame_count, frame_time)

        cv2.polylines(resized_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow('Frame', resized_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

        frame_count += 1  # Frame sayısını artır

    cap.release()
    cv2.destroyAllWindows()

def draw_keypoints(boxes, keypoints, frame, img_height, img_width, points, video_time):
    in_roi = False  # Başlangıçta in_roi değerini False olarak belirleyin
    for i in range(len(boxes)):
        box = boxes.xywh[i].tolist()
        x_center, y_center, width, height = box
        x_min = int(max(0, (x_center - width / 2) * img_width))
        y_min = int(max(0, (y_center - height / 2) * img_height))
        x_max = int(min(img_width, (x_center + width / 2) * img_width))
        y_max = int(min(img_height, (y_center + height / 2) * img_height))

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        keypoints_person = keypoints[i].data.tolist()

        for kp in keypoints_person:
            px = int(min(max(0, kp[0]), img_width))
            py = int(min(max(0, kp[1]), img_height))

            visibility = kp[2]
            color = (0, 0, 255) if visibility < 0.5 else (0, 255, 0)
            cv2.circle(frame, (px, py), 5, color, -1)

            if cv2.pointPolygonTest(points, (px, py), False) >= 0:
                in_roi = True
                break

    # CSV yazımı, in_roi değerinin False veya True olması fark etmeksizin gerçekleşir
    with open('output.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([round(video_time, 2), in_roi])  # Zaman damgasını iki ondalık basamakla yaz

    print(f"Elapsed Video Time: {round(video_time, 2)}s, In ROI: {in_roi}")

process_videos(video_path, model, resized_points, original_points, scale_factor)
