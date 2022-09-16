from modules.module_embedding import audio_Wav2Vec2, video_resnet50
import argparse
import librosa
import os
import pandas as pd
import numpy as np
from PIL import Image
# import cv2





def get_args(description='data embedding'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--wav2vec2_base_960h", default="/home/lishuai/pretrainedmodel/wav2vec2-base-960h", help="pretrained wav2vect2.0 path")
    parser.add_argument("--resnet50", default="/home/lishuai/pretrainedmodel/resnet-50", help="pretrained resnet50 path")
    parser.add_argument("--cmumosei_ids_path",default="/home/lishuai/Dataset/CMU-MOSEI-RAW/processed_data/ids.csv")
    parser.add_argument("--cmumosei_audio_path",default="/home/lishuai/Dataset/CMU-MOSEI-RAW/processed_data/audio/WAV_fromVideo")
    parser.add_argument("--cmumosei_video_path",default="/home/lishuai/Dataset/CMU-MOSEI-RAW/processed_data/video/version_img_size_224_img_scale_1.3")
    parser.add_argument("--csv_path",default="/home/lishuai/workspace/data/cmumosei.csv")
    parser.add_argument("--feature_path", default="/home/lishuai/workspace/feature/cmumosei")

    args = parser.parse_args()

    return args


def cmumosei_data_embedding(opts):
    ids_path = opts.cmumosei_ids_path
    audio_path = opts.cmumosei_audio_path
    video_path = opts.cmumosei_video_path
    ids = np.array(pd.read_csv(ids_path))
    ids = ids.reshape(ids.shape[0], ).tolist()
    data = []
    for id in ids:
        print("id:", id)
        wave_data, samplerate = librosa.load(os.path.join(audio_path, id + ".wav"), sr=16000)
        audioFeature = audio_Wav2Vec2(opts,wave_data)
        videodir = os.path.join(video_path, id + "_aligned")
        imglist = os.listdir(videodir)
        video = []
        for image in imglist:
            imgpath = os.path.join(videodir, image)
            img = Image.open(imgpath)
            video.append(np.array(img))
            img.close()
        videoFeature = video_resnet50(opts,video)
        audio_file = "audio/"+id+".npy"
        video_file = "video/"+id+".npy"
        data.append([audio_file,video_file,0])
        audioFeaturePath = os.path.join(opts.feature_path,audio_file)
        videoFeaturePath = os.path.join(opts.feature_path,video_file)
        np.save(audioFeaturePath,audioFeature)
        np.save(videoFeaturePath,videoFeature)
    df = pd.DataFrame(data,columns=["audio_feature","video_feature","emotion_label"])
    df.to_csv(opts.csv_path,)
    print("cmumosei_data_embedding done")
if __name__=="__main__":
    args = get_args()
    cmumosei_data_embedding(args)

