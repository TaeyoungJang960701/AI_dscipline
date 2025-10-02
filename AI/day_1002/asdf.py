import whisper

# 모델 불러오기 (기본 small, 더 정확하려면 "medium"이나 "large")
model = whisper.load_model("small")

# mp3 파일 불러와서 전사
result = model.transcribe("1001_topic.wav", language="ko")

# 전사된 텍스트 출력
print(result["text"])
