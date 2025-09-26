from django.shortcuts import render

# Create your views here.
import os
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import tensorflow as tf

# ----------------------------------------------------------------
# 앱 시작 시 모델을 한 번만 로드 (비용 절약)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'CNN_model', 'food101_saved')  # 맞춰서 경로 설정
model = tf.keras.models.load_model(MODEL_PATH)
# 클래스 이름 리스트 (training시 사용한 ds_info에서 복사)
CLASS_NAMES = [
    # food101 class list (101 names). 예시로 앞 몇개만 적음 — 실제로는 전체 101개 넣으세요.
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beignets",
    # ... 전체 101개
]
IMG_SIZE = 160  # 학습시 사용한 이미지 사이즈 (여기선 160)

# ----------------------------------------------------------------
def index(request):
    return render(request, 'classifier/index.html')

# 업로드 & 예측 뷰 (AJAX)
@csrf_exempt
def predict_image(request):
    if request.method != 'POST':
        return JsonResponse({'error':'POST만 지원'}, status=400)

    if 'file' not in request.FILES:
        return JsonResponse({'error':'file 파라미터 없음'}, status=400)

    # 1) 업로드 파일 저장
    upload = request.FILES['file']
    upload_name = upload.name
    save_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, upload_name)

    with open(save_path, 'wb') as f:
        for chunk in upload.chunks():
            f.write(chunk)

    # 2) 이미지 열기 + 전처리 (학습때와 동일한 전처리)
    img = Image.open(save_path).convert('RGB')
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype(np.float32)
    # 학습에서 (image/127.5)-1.0 을 사용하셨으니 동일하게
    img_array = (img_array / 127.5) - 1.0
    input_batch = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)

    # 3) 예측
    preds = model.predict(input_batch)  # shape (1,101)
    probs = preds[0]
    top_idx = np.argmax(probs)
    top_conf = float(probs[top_idx])
    top_label = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)

    # 4) 시각화: 원본 이미지 + 확률 바 차트 -> 파일로 저장 (선택)
    viz_dir = os.path.join(settings.MEDIA_ROOT, 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    viz_path = os.path.join(viz_dir, f'viz_{os.path.splitext(upload_name)[0]}.png')

    # 상위 5개 라벨 시각화
    top_k = 5
    top_k_idx = probs.argsort()[-top_k:][::-1]
    top_k_labels = [CLASS_NAMES[i] for i in top_k_idx]
    top_k_probs = probs[top_k_idx]

    plt.figure(figsize=(6,4))
    plt.subplot(1,2,1)
    plt.imshow(img_resized)
    plt.axis('off')
    plt.title(top_label)

    plt.subplot(1,2,2)
    y_pos = np.arange(top_k)
    plt.barh(y_pos[::-1], top_k_probs[::-1], align='center')  # 역순으로 보이게
    plt.yticks(y_pos, top_k_labels)
    plt.xlabel('probability')
    plt.tight_layout()
    plt.savefig(viz_path)
    plt.close()

    viz_url = os.path.join(settings.MEDIA_URL, 'viz', os.path.basename(viz_path))

    response = {
        'pred_label': top_label,
        'confidence': top_conf,
        'viz_url': viz_url,
    }
    return JsonResponse(response)
