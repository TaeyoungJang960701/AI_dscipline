import os, json, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet_v2
import sys

IMG_SIZE = (224,224)

def load_class_names(path = 'class_name.txt'):
    with open(path, 'r', encoding = 'utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    return names

# 모델이 기대하는 입력차원과 형식을 맞춰주는 전처리단계 함수
def load_and_preprocess(img_path):
    img = tf.keras.utils.load_img(img_path, target_size = IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)      # 이미지를 float32 배열(0 ~ 255)로 변환
    arr = np.expand_dims(arr, axis = 0)         # 배열 차원을 추가  (224, 224, 3) -> (1, 224, 224, 3)
    return arr

def main():
    if len(sys.argv) < 2:
        print('분류할 파일명.확장자 입력')
        sys.exit(1)
    
    image_path = sys.argv[1]
    print(image_path)

    # 이미지 분류 모델 로딩
    model = keras.models.load_model(
        'best_model.keras',
        compile = False,
        custom_objects = {'preprocess_input':mobilenet_v2.preprocess_input},
    )

    # 'class_name.txt'를 읽어 인덱스 -> 클래스명과 맵핑
    # class_names = load_class_names(r'C:\Users\acorn\OneDrive\Desktop\Artificial Intelligence.class_name.txt')

    # class_names = load_class_names('class_name.txt')

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    class_names = load_class_names(os.path.join(BASE_DIR, 'C:/Users/acorn/OneDrive/Desktop/Artificial Intelligence/anal6/day_0926/class_name.txt'))
    print(class_names)

    # 입력 이미지 전처리
    x = load_and_preprocess(image_path)
    preds = model.predict(x, verbose = 0)[0]
    print(preds)
    top_idx = int(np.argmax(preds))
    top_prob = float(preds[top_idx])

    # 판정 확률 1위에서 3위까지 출력
    print(f'예측값 : {class_names[top_idx]} (확률 : {top_prob:.3f})')

    order = np.argsort(-preds)  
    # 소트 함수 내부 변수에 -를 붙이면 내림차순으로 변해 원래는 오름차순인가봐

    print('분류 예측 결과 : ')
    for i in order[:3]:
        print(f'{class_names[i]:}  {preds[i]:.3f}')
    
if __name__ == '__main__':
    main()

# 실행은 이렇게 할거야 프롬트창에서
# > python classify.py 파일명.jpg