# (선택) !pip install ultralytics opencv-python

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 일부 환경에서 OMP/KMP 중복 로드 오류 회피용(경고 억제)

import subprocess
import sys

try:
    # ultralytics 패키지(내부에 YOLO 클래스 포함) 임포트 시도
    from ultralytics import YOLO
except ModuleNotFoundError:
    print('ultralytics가 설치 되지 않아 설치를 시작합니다.')
    try:
        # 현재 파이썬 인터프리터로 pip 실행 → ultralytics 설치
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics'])
    except subprocess.CalledProcessError as e:
        # 설치 실패 시 종료 안내
        raise SystemExit('ultralytics 설치 실패, 수동으로 설치하세요')
    # 설치 완료 후 재임포트
    from ultralytics import YOLO

import ultralytics
ultralytics.checks()    # 환경 점검(패키지/권한/CUDA 상태 등). 이력과 경고 출력

try:
    # COCO(80 클래스)로 학습된 YOLOv8n 가중치 로드
    # 최초 호출 시 인터넷에서 .pt 파일 자동 다운로드
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f'error loading model: {e}')

# 클래스 ID → 클래스명 매핑(dict) 출력 (예: {0:'person', 1:'bicycle', ...})
print(model.names)      # COCO dataset(class 80개)

# 클래스 개수 출력(보통 80)
print(len(model.names))

# 이미지 로딩 및 표시
from PIL import Image
import matplotlib.pyplot as plt

image_path = 'C:/Users/acorn/OneDrive/Desktop/Artificial Intelligence/AI/day_0929/yoloex/dog.jpg'  # 감지 테스트할 이미지 경로

try:
    image = Image.open(image_path)  # PIL 이미지 열기
    plt.imshow(image)               # 원본 미리보기
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f'error : {e}')
    exit()
# 추론 실행 (이미지 경로 또는 배열/PIL 이미지 모두 가능)
results = model(image_path)  # 또는 model(image) 도 가능

# 결과(첫 프레임)에서 주석(render)된 이미지를 가져옴 (BGR ndarray)
annot_bgr = results[0].plot()

# matplotlib는 RGB를 기대하므로 BGR → RGB 변환
annot_rgb = annot_bgr[..., ::-1]

plt.figure(figsize=(8, 6))
plt.imshow(annot_rgb)
plt.title('YOLOv8n detection')
plt.axis('off')
plt.show()

# (선택) 감지된 박스/클래스/점수 텍스트로 확인
for box in results[0].boxes:
    cls_id = int(box.cls[0].item())
    score = float(box.conf[0].item())
    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
    print(f"class={model.names[cls_id]} ({cls_id}), conf={score:.2f}, box={xyxy}")