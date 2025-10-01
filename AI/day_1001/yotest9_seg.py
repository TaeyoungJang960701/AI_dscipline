# 인스턴스 세그멘테이션 : 욜로가 직접 내주는 결과. 객체마다 마스크가 따로 존재
# 의미론적 세그멘테이션 : 이미지 내의 픽셀단위로 '이 픽셀은 어느 클래스에 속한다' 만 표현함

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'OK'

import cv2, numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

IMG_PATH = 'images/animals.jpg'
OUT_DIR = 'day_1001/seg_out2'
os.makedirs(OUT_DIR, exist_ok = True)

model = YOLO('day_1001/yolov8n-seg.pt')

im_bgr = cv2.imread(IMG_PATH)

H, W = im_bgr.shape[:2]

res = model(im_bgr, verbose = False)[0]
annotated = res.plot()
cv2.imwrite(os.path.join(OUT_DIR, 'day_1001/seg_result.jpg'), annotated)

# pytorch tensor -> numpy 배열로 변환
has_masks = (res.masks is not None)

if has_masks:
    masks_np = res.masks.data.cpu().numpy()                 # 객체별 픽셀 마스크 shape = (N, H, W)
    boxes_np = res.boxes.xyxy.cpu().numpy().astype(int)     # 경계박스 좌표
    confs_np = res.boxes.conf.cpu().numpy()                 # 신뢰도 점수
    classes_np = res.boxes.cls.cpu().numpy().astype(int)    # 클래스 id(번호)


else:
    masks_np = boxes_np = confs_np = classes_np = None

# 마스크 오버레이
overlay = im_bgr.copy()
if has_masks:
    for m in masks_np:
        m_bin = cv2.resize(m, (W,H), interpolation = cv2.INTER_NEAREST) > 0.5
        color_mask = np.zeros_like(overlay)     # 이건 제로스로 0으로 맞춰놔서 아마 다 깜깜할거야
        color_mask[m_bin] = (0, 255, 0)     # 객체 마스크 픽셀만 초록색으로 채움
        overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.4, 0.0)
        # 원본 + 컬러 마스크

cv2.imwrite(os.path.join(OUT_DIR, 'seg_overlay.jpg'), overlay)

# 객체별 배경 제거
crops_dir = os.path.join(OUT_DIR, 'seg_drops')
os.makedirs(crops_dir, exist_ok = True)

if has_masks and len(masks_np) > 0 :
    masks_full = np.stack([
        cv2.resize(m, (W, H), cv2.INTER_NEAREST) > 0.5 for m in masks_np
    ], axis = 0)

    # 탐지된 객체의 배경을 제거해 png 파일로 잘라내기
    for i, (m_full, box, cls_id, conf) in enumerate(zip(masks_full, boxes_np, classes_np, confs_np)):
        x1,y1,x2,y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)     # 좌상단 좌표가 이미지 밖으로 나가면 
        x2, y2 = max(W, x2), max(H, y2)     # 우하단 좌표가 이미지 밖이면 0으로 보정
        if x2 <= x1 or y2 <= y1:
            continue

        crop_bgr = im_bgr[y1:y2, x1:x2]
        crop_mask = (m_full[y1:y2, x1:x2] * 255).astype(np.uint8)
        crop_bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)   # BGR -> BGRA(alpha 채널 추가)
        crop_bgra[..., 3] = crop_mask   # 알파 채널에 마스크가 적용됨 -> 배경은 투명해지고 객체는 불투명해짐
        
        # 클래스 이름 또는 id 얻기
        name = model.names[int(cls_id)] if hasattr(model, 'names') else str(cls_id)
        cv2.imwrite(os.path.join(crops_dir, f'crop_{i}_{name}_{conf:.2f}.png'), crop_bgra)
        
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
plt.close()

# 의미론적 분할(semantic segmentation)
sem_canvas = np.zeros((H, W, 3), dtype=np.uint8)      # 최종 색상 이미지
conf_map = np.zeros((H, W), dtype=np.float32)         # 선택된 인스턴스의 신뢰도를 기록할 맵

def class_color(c:int):
    return((23 * c) % 256, (19 * c) % 256, (77 * c) % 256)

if has_masks and len(masks_np) > 0 :
    for m_full, cls_id, conf in zip(masks_full, classes_np, confs_np):
        color = class_color(int(cls_id))
        update = m_full & (conf > conf_map)
        sem_canvas[update] = color
        conf_map[update] = conf

cv2.imwrite(os.path.join(OUT_DIR, 'seg_semantic.png'), sem_canvas)