import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 일부 환경에서 OMP/KMP 중복 로드 오류

import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt


model = YOLO('yolov8n.pt')

image_path='C:/Users/acorn/OneDrive/Desktop/Artificial Intelligence/AI/day_0929/yoloex/images.jpg'

try:
    image=cv2.imread(image_path)
except FileNotFoundError as e:
    print('에러 : ',e)
    raise SystemExit

original = image.copy()

results=model(image)
# print(results)

person_count=0

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = box.conf[0].item()

        if label.lower()=='person':
            person_count+=1

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0),2)
        cv2.putText(image, f'{label}:{confidence:.2f}', (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print(f'감지된 사람의 수 : {person_count}명')

plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(f'Detected person : {person_count}')
plt.show()

# 바운딩 박스로 된 이미지 전체를 저장
out_path='yoloex/yotest3_out.jpg'
cv2.imwrite(out_path,image)
print('바운딩 박스로 된 이미지 전체를 저장 완료')

# 바운딩 박스 내부 객체만 저장
os.makedirs('crops', exist_ok = True)
for idx, result in enumerate(results):
    for j, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = box.conf[0].item()

        # 원본 이미지에서 ROI(Region or Interest, 관심영역) 추출

        # cropped = image[y1:y2, x1:x2]   
        cropped = original[y1:y2, x1:x2]

        # image(H, W, 3(컬러의 채널 얘기야)).
        # 배열 슬라이싱을 통해 선택된 작은 이미지 배열을 반환함
        # print('cropped : ', cropped)

        # 선택된 이미지 배열 저장
        crop_path = os.path.join('crops', f'C:/Users/acorn/OneDrive/Desktop/Artificial Intelligence/AI/day_0929/yoloex/crop_{idx}_{j}_{label}_{confidence:.2f}.jpg')
        cv2.imwrite(crop_path, cropped)

        print(f'객체 {label}이 {crop_path}에 저장됨')

# 감지된 객체의 중심 좌표
p_count = 0

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = box.conf[0].item()

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        if label.lower() == 'person':
            p_count += 1
            print(f'Person -> {p_count} : 중심좌표는 ({center_x}, {center_y}), 신뢰도 : {confidence:.2f}')
            # 중심점 그려놓기
            cv2.circle(image, (center_x, center_y), 5, (0,0,255), -1)
            coord_text = f'({center_x}, {center_y})'
            cv2.putText(image, coord_text, (center_x + 10, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{label} : {confidence:.2f}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

plt.figure(figsize = (10, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()