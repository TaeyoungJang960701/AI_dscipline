# 웹캠을 통해 카메라가 정상적으로 작동 하는지 확인하기
# 웹사이트 (webcamtests.com)에서 카메라 정상 작동 확인 가능
# Python으로 아래 코드를 실행하면 웹캠을 통해 실시간으로 객체를 감지할 수 있다.
"""
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0번 카메라(기본 웹캠)

if not cap.isOpened():
    print("웹캠을 열 수 없네 ㅠㅠ.")
else:
    print("웹캠이 열렸습니다. ESC 눌러 종료하세요.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
"""
import cv2
from ultralytics import YOLO
import time
import os

model = YOLO('yolov8n.pt')
# print(model.names)

# 감지된 이미지 저장 폴더
save_dir = 'yoloex/test2_dir'
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('웹캠 사용불가')
    exit()
else:
    print('웹켐 사용가능')

cv2.namedWindow('YOLO 실시간 객체 감지', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLO 실시간 객체 감지', 800, 600)

# 중복 저장 방지(3초 내에는 같은 객체 저장 X)
last_saved_time = {}

while True:
    ret, frame = cap.read()    # ret:프레임 읽기 성공/실패(T/F), frame
    if not ret:
        print('프레임을 읽을 수 없음')
        break

    results = model(frame, verbose=False)  # model.predict(기본값 변경)

    # 특정 객체만 감지에 참여
    allowed_labels = [
        'person', 'laptop', 'keyboard', 'cell phone', 'book'  # COCO 라벨 표기
    ]

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        # (수정) box를 사용하기 전에 for문으로 순회
        for box in boxes:
            # 좌표 추출 (텐서를 안전하게 숫자로 변환)
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            label = result.names[int(box.cls[0])]
            confidence = float(box.conf[0].item())

            # 라벨 필터
            if label not in allowed_labels:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
            cv2.putText(frame, f'{label}:{confidence:.2f}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 3초 간격으로 중복 방지 저장
            now = time.time()
            last_time = last_saved_time.get(label, 0)

            if now - last_time >= 3:
                filename = f'{label}_{int(now)}.jpg'   # 확장자 추가 유지
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, frame)
                print(f'저장성공 : {filepath}')
                last_saved_time[label] = now

    # 감지된 프레임 화면에 출력 (창 이름 일치)
    cv2.imshow('YOLO 실시간 객체 감지', frame)

    key = cv2.waitKey(1)      # 1ms동안 입력 대기 , 아무키도안누르면 -1 반환

    if key != -1:
        print('눌린 키:', key, chr(key))

    if (key & 0xFF) == 27:
        break

# 자원정리
cap.release()           # 사용중인 카메라 장치 해제
cv2.destroyAllWindows()