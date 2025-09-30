# 이미지 디텍션 + TTS(Text To Speach)
# 유기동물 사진을 제출하면 탐지 후 안내소로 안내하는 문구를 소리로 표현

# pip install playsound==1.2.2
# pip install gTTS

from gtts import gTTS
from IPython.display import Audio   # Jupyter notebook의 경우엔 여기서 소리가 재생
from playsound import playsound     # 학원컴 비주얼 스튜디오(local)에선 이거로 소리 재생

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

""" 
def speak_shelter_info(message): 
tts = gTTS(text=message, lang='ko') 
tts.save('sound.mp3') 
영어 파일명으로 저장 
try: # playsound('sound.mp3') 
except Exception as e: 
print('오디오 출력장치가 없거나 playsound 오류:', e) 
message = '"성공은 실패를 거듭해도 열정을 잃지 않는 것이다." – 윈스턴 처칠' 
speak_shelter_info(message) 
"""



def show_shelter_info_func(region, shelters, detected_info):
    # shelters를 함수처럼 호출한 부분 수정 → dict 조회(지피티)
    shelter_info = shelters.get(region, shelters['기본'])
    
    # join 구문 및 message tuple → 문자열로 수정(지피티)
    pet_summary = f"{detected_info['count']}마리 ({','.join(detected_info['labels'])})"
    message = (
        f"유기동물 탐지 결과 : \n"
        f"- 탐지된 동물 수 : {detected_info['count']}\n"
        f"- 종류 : {', '.join(detected_info['labels'])}\n\n"
        f"{region} 지역 보호소 정보 : \n{shelter_info}"
    )
    print('보호소 정보 : ')
    print(message)      # 문자 안내

    # 음성 안내
    try:
        tts = gTTS(text = f"{region} 지역에 유기된 {pet_summary}가 감지되었습니다.\
                   가까운 보호소는 {shelter_info} 입니다", lang = 'ko')
        tts.save('day_0930/yt5sound.mp3')
        playsound('day_0930/yt5sound.mp3')

        # tts는 띄어쓰기를 인정해서 띄어쓰기 해주면 선명하게 잘 들린대

    except Exception as e:
        print(f'음성안내 실패 원인 : {type(e).__name__} - {e}')


def handle_stray_pet_func(region, shelters, detected_info):
    print('유기 동물로 추정됩니다')
    show_shelter_info_func(region, shelters, detected_info)


"""
region = '강남'
shelters = {    # 보호소 정보
    '서울' : '서울 반려동물 보호센터 : 02-1234-5678',
    '기본' : '전국 반려동물 보호연합 : 1577-1000',
}

detected_info = {
    'count' : 3,
    'labels' : ['호랑이','사자','코끼리']
}

handle_stray_pet_func(region, shelters, detected_info)  
"""

# 탐지 정보 로그 파일로 저장
def save_detection_log_func(image_path, detection_data):
    log_file_name = 'day_0930/yotest5detec.txt'
    now = datetime.now().strftime('%Y-%m%d %H:%H:%S')

    with open(log_file_name, 'a', encoding = 'utf-8') as f:
        f.write(f'\n[{now}] 이미지 : {image_path}\n')
        f.write(f'탐지된 객체 수 : {len(detection_data)}\n')
        for d in detection_data:
            f.write(f" - {d['label']} : box = {d['box']}, confidence = {d['confidence']:.2f}\n")
        
        f.write("-" * 40 + '\n')
    print(f'탐지결과가 {log_file_name}에 저장됨')


# 유기동물 감지 함수
def detect_pets_func(image_path):
    pet_desc = {
        'dog' : '댕댕이',
        'cat' : '고양이',
    }

    shelters = {    # 보호소 정보
        '서울' : '서울 반려동물 보호센터 : 02-1234-5678',
        '부산' : '부산 유기동물 보호소 : 051-9999-7878', 
        '기본' : '전국 반려동물 보호연합 : 1577-1000',
    }
    stray_keywords = ['street', 'road', 'outside', 'stray']

    model = YOLO('day_0930/yolov8n.pt')
    image = cv2.imread(image_path)
    if image is None:
        print('이미지를 불러올 수 없어요~')
        return
    
    results = model(image)
    detected_pets = []
    detection_data = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0].item()

            if label in pet_desc:
                detected_pets.append(label)
                detection_data.append({
                    'label' : pet_desc[label],
                    'box'   : (x1,y1,x2,y2),
                    'confidence' : confidence 
                })
                cv2.rectangle(image, (x1,y1), (x2, y2), (0,255, 0), 2)
                cv2.putText(image, f'{label} : {confidence:.2f}', (x1, y2 - 10),\
                            cv2.FONT_HERSHEY_SIMPLEX, -.5, (0,255,0),2)
                
    # 결과 이미지 저장
    output_path = 'day_0930/outtest5.jpg'   # 경로만 수정해주세요
    cv2.imwrite(output_path, image)
    print(f'감지된 이미지 파일로 저장 성공 : {output_path}')

    # 이미지 보기
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    plt.close()

    if detected_pets:   # detected_pets 리스트에 감지된 동물이 있으면 if 함수가 실행됨
        print('감지된 동물 결과 : ')
        for pet in set(detected_pets):
            print(f'- {pet_desc.get(pet, pet)}')
        
        # 감지된 정보들을 텍스트 파일로 저장
        save_detection_log_func(image_path, detection_data)

        # 유기동물
        # 이미지 경로에 stray_keywords에 추가등록된 단어가 있는 경우
        if any(pet in['dog','cat'] for pet in detected_pets) and \
            any(keyword in image_path.lower() for keyword in stray_keywords):
            detected_info = {
                'count' : len(detection_data),
                'labels' : sorted(set([d['label'] for d in detection_data]))
            }
            handle_stray_pet_func(region = '서울', 
                                  shelters = shelters, detected_info = detected_info)
    else:
        print('유기동물이 감지되지 않았습니다.')
        

detect_pets_func('day_0929/yoloex/street_cat.jpg')