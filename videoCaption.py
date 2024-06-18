from bark import SAMPLE_RATE, generate_audio, preload_models
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from scipy.io.wavfile import write as write_wav
import moviepy.editor as mp
import cv2
import os

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

preload_models()

interval = 10

# 동영상 파일 경로
video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)

# 프레임 추출 및 캡셔닝
captions = []
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = frame_rate * interval  # interval초마다 한 프레임씩 캡션 생성

while cap.isOpened():
    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_id % frame_interval == 0:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        
        generated_ids = model.generate(pixel_values)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        captions.append(generated_text)
        
cap.release()

print("caption length: " + str(len(captions)))

# 캡션을 각각 음성으로 변환하고 저장
audio_files = []
for i, caption in enumerate(captions):
    audio_array = generate_audio(caption, history_prompt="v2/en_speaker_6")
    audio_file = f"audio_{i}.wav"
    write_wav(audio_file, SAMPLE_RATE, audio_array)
    audio_files.append(audio_file)

# 동영상 불러오기
video_clip = mp.VideoFileClip(video_path)

# 각 캡션의 오디오 클립을 생성하고 해당 위치에 추가
audio_clips = []
start_time = 0
for i, audio_file in enumerate(audio_files):
    audio_clip = mp.AudioFileClip(audio_file)
    audio_clips.append(audio_clip.set_start(start_time, change_end=False))
    start_time += interval

# 전체 오디오 클립을 결합
final_audio = mp.concatenate_audioclips(audio_clips)

# 동영상에 오디오 추가
final_clip = video_clip.set_audio(final_audio)
final_clip.write_videofile("final_output_video.mp4", codec="libx264")

# 임시 오디오 파일 삭제
for audio_file in audio_files:
    os.remove(audio_file)