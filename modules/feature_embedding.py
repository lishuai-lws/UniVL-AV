from torch import nn
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import argparse
import librosa
import os
import pandas as pd
import numpy as np
from PIL import Image



class AudioWav2Vec2(nn.Module):
    def __init__(self,modelpath):
        super().__init__()
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(modelpath,padding=True)
        self.model = Wav2Vec2Model.from_pretrained(modelpath)

    def forward(self, wavdata):
        data = self.tokenizer(wavdata, return_tensors="pt").input_values
        feature = self.model(data)
        return feature

class ResNet50(nn.Module):
    def __init__(self, modelpath):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(modelpath)
        self.model = ResNetForImageClassification.from_pretrained(modelpath)

    def forward(self, image):
        inputs = self.feature_extractor(image, return_tensors="pt")
        feature = self.model(**inputs)
        return feature

def get_args(description='data embedding'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")


    args = parser.parse_args()

    if args.sampled_use_mil:  # sample from each video, has a higher priority than use_mil.
        args.use_mil = True

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_pretrain:
        raise ValueError("`do_pretrain` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    args.checkpoint_model = '{}_{}_{}_{}.checkpoint'.format(args.checkpoint_model, args.bert_model, args.max_words,
                                                            args.max_frames)

    return args

def audio_Wav2Vec2(opts, wavdata):
    modelpath = opts.wav2vec2_base_960h
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(modelpath, padding=True)
    model = Wav2Vec2Model.from_pretrained(modelpath)
    input_values = tokenizer(wavdata, return_tensors="pt").input_values
    feature = model(input_values)
    return feature
def video_resnet50(opts, images):
    modelpath = opts.resnet50
    feature_extractor = AutoFeatureExtractor.from_pretrained(modelpath)
    model = ResNetForImageClassification.from_pretrained(modelpath)
    features = []
    for image in images:
        inputs = feature_extractor(image, return_tensors="pt")
        feature = model(**inputs)
        features = features.append(feature)
    return features
def cmumosei_data_embedding(opts):
    ids_path = opts.cmumosei_ids_path
    audio_path = opts.cmumosei_audio_path
    video_path =opts.cmumosei_video_path
    ids = np.array(pd.read_csv(ids_path))
    ids = ids.reshape(ids.shape[0], ).tolist()
    print(ids[:5])
    for id in ids[:5]:
        print("id:", id)
        wave_data, samplerate = librosa.load(os.path.join(audio_path, id + ".wav"))
        audioFeature = audio_Wav2Vec2(opts,wave_data)
        videodir = os.path.join(video_path, id + "_aligned")
        imglist = os.listdir(videodir)
        video = []
        for image in imglist:
            imgpath = os.path.join(videodir, image)
            img = Image.open(imgpath)
            video.append(np.array(img.getdata()))
            img.close()
        videoFeature = video_resnet50(opts,video)
        audioFeaturePath = os.path.join(opts.cmumosei_audio_feature_path,id,".npy")
        videoFeaturePath = os.path.join(opts.cmumosei_video_feature_path,id,".npy")
        np.save(audioFeaturePath,audioFeature)
        np.save(videoFeaturePath,videoFeature)
if __name__=="__main__":
    args = get_args()
    cmumosei_data_embedding(args)

