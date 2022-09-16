import sys
sys.path.append("/home/lishuai/UniVL-AV/modules")
from modules.module_embedding import audio_Wav2Vec2, video_resnet50
model = audio_Wav2Vec2()
print("helloworld")