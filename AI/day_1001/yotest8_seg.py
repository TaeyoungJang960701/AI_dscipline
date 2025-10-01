# 이미지 segmentation
import os, cv2, numpy as np
from ultralytics import YOLO

IMG_PATH = 'images/image1.jpg'
OUT_DIR = 'day_1001/seg_out'
os.makedirs(OUT_DIR, exist_ok = True)

im = cv2.imread(IMG_PATH)
assert im is not None, f'이미지 읽기 실패 : {IMG_PATH}'

H, W = im.shape[:2]     # (175, 287, 3)
print(H, W)

# model
model = YOLO('day_1001/yolov8n-seg.pt')
res = model(im)[0]
# print(res)

cv2.imwrite(os.path.join(OUT_DIR, 'anno1.jpg'), res.plot())
# (1, 3, 416, 640)

# res.plot() : 
# 원본 이미지 위에 바운딩박스, 레이블, confidence(신뢰도 점수), segmentaion mask를
# 한번에 그려서 BGR 이미지로 반환해줌

# 마스크가 없으면 작업 종료
if res.masks is None or len(res.masks.data) == 0:
    print('마스크가 없어요 ㅜㅜ')
    raise SystemExit

m_small = res.masks.data.cpu().numpy()
# print(m_small)

masks = np.stack([
    cv2.resize(m, (W,H), cv2.INTER_NEAREST) > 0.5 for m in m_small
], axis = 0)    
# 각 객체별 (H, W) 마스크를 모아 (N, H, W) 배열로 만듦
# N개의 bool 마스크 스택

# print(masks)    # [[[False False False ... False False False]

# segmentation 전단계 : mask preview
# 마스크가 같은 위치 픽셀에 대해 객체중 하나라도 True(1)이면 N개 마스크를 OR 연산으로 합침
mask_union = (masks.any(axis = 0).astype(np.uint) * 255)
cv2.imwrite(os.path.join(OUT_DIR, 'mask_preview.png'), mask_union)

# 최종 segmentation : 컬러 오버레이 + 외곽선
def color(i):
    return ((37 * i) % 256, (17 * i) % 256, (91 * i) % 256)

final = im.copy()   # 직접 원본에 덧칠하지 않고 안전하게 복사본에서 작업
blend = np.zeros_like(im)   # 오버레이 색 채우기 캔버스

# 컬러 오버레이(blend)는 객체 내부를 색칠하고 경계선을 그리는 두가지 작업
for i, m in enumerate(masks):
    blend[m] = color(i)
    cnts, _ = cv2.findContours(     # 마스크 외곽선 자료 추출
        # findContours() 클래스가 masks의 외곽선 좌표를 추출해준다
        (m.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        # 0/1(이진법) -> 0 ~ 255 이진화    가장 바깥쪽 외곽선만    꼭짓점 단순화    
        )
    cv2.drawContours(final, cnts, -1, (255, 255, 255), 2, cv2.LINE_AA) # AA = Anti Aliasing

# 반투명 합성
final = cv2.addWeighted(final, 1.0, blend, 0.45, 0.0)   # open cv가 갖고 있는 기능들이 아주 많다
# addWeighted(첫번째 입력 이미지, 첫번째 이미지에 가해지는 가중치, 두번째 입력 이미지, 두번째에 가해지는 가중치, 모든 픽셀에 더해지는 상수(밝기 조정용))

cv2.imwrite(os.path.join(OUT_DIR, 'final_preview.jpg'), final)

cv2.imshow('final segmentation', final)
cv2.waitKey(0)
cv2.destroyAllWindows()