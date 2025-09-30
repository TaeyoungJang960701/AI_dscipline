# YOLO 탐지 결과를 이미지별로 정리해서 CSV 파일로 저장한다.
# 이어서 CSV를 읽어 DataFrame에 담아 어쩌구 저꺼구


# images 폴더를 만들어 image1,2,3을 넣어놓고 실습
import os
import pandas as pd
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
img_dir = 'images'

# 해당 확장자를 끌어와 list에 저장하는 문구
img_paths = [os.path.join(img_dir, f) 
            for f in os.listdir(img_dir) 
            if f.lower().endswith(('.jpg', '.jpeg','.png'))]

print(img_paths)

records = []

for path in img_paths:
    results = model(path, conf = 0.25, verbose = False)[0]      
    # confidence가 0.25퍼 이상 되는애들만 뽑아줘
    boxes = results.boxes
    names = results.names
    # print(boxes, names)
    if len(boxes) == 0:
        records.append({
            'image' : os.path.basename(path),
            'object_count' : 0,
            'classes' : '',
            'avg_confidence' : 0.06
        })
        continue

    cls_ids = boxes.cls.cpu().numpy().astype(int)
    print(cls_ids)
    
    confs = boxes.conf.cpu().numpy()
    print(confs)
    
    classes = [names[i] for i in cls_ids]
    print(classes)
    avg_conf = float(confs.mean())

    records.append({
            'image' : os.path.basename(path),
            'object_count' : len(cls_ids),
            'classes' : ','.join(sorted(set(classes))),
            'avg_confidence' : round(avg_conf, 3)
        })
    
# records -> DataFrame -> csv
df = pd.DataFrame(records)
print(df)
df.to_csv('day_0930/yotest6report.csv', index = False, encoding = 'utf-8-sig')
print('csv 저장 완료')

mydf = pd.read_csv('day_0930/yotest6report.csv')
num_images = len(df)
total_objects = df['object_count'].sum()
print('total_objects : ', total_objects)

# 전체 신뢰도 평균
overall_avg_conf = df.loc[df['avg_confidence'] > 0, 'avg_confidence'].mean() \
    if total_objects > 0 else 0.0
    
# 클래스별 등장 빈도
class_counts = {}
for cls_str in df['classes']:
    if cls_str:
        for c in cls_str.split(','):
            class_counts[c] = class_counts.get(c, 0) + 1

print('YOLO detection summary')
print(f'총 이미지 수       : {num_images}')
print(f'총 탐지 객체 수    : {total_objects}')
print(f'전체 신뢰도 평균    : {overall_avg_conf:.4f}')
print(f'클래스별 등장 횟수  : ')
for k, v in class_counts.items():
    print(f'    {k} : {v}')