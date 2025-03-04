import librosa
import math
import os
import numpy as np
import random
import torch
import pickle

from .stream_pipeline_offline import StreamSDK


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def ditto_run(SDK: StreamSDK,
              audio,
              source: str | list,
              more_kwargs: str | dict = {}):
    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    SDK.setup(source, "", **setup_kwargs)
    if isinstance(audio, str):
        audio, sr = librosa.core.load(audio, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)

    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    aud_feat = SDK.wav2feat.wav2feat(audio)
    SDK.audio2motion_queue.put(aud_feat)

    SDK.close()

    video_frame_list = SDK.getVideoFrames()
    SDK.clearVideoFrames()
    return video_frame_list


if __name__ == "__main__":

    # init sdk
    data_root = "../models/ditto/ditto_trt_Ampere_Plus"  # model dir
    cfg_pkl = "../models/ditto/ditto_cfg/v0.4_hubert_cfg_trt.pkl"  # cfg pkl
    SDK = StreamSDK(cfg_pkl, data_root)

    # input args
    audio_path = "../example/audio.wav"   # .wav
    source_path = "../example/image.png"  # video|image
    output_path = "../example/output.mp4"  # .mp4

    # run
    # seed_everything(1024)
    # audio, sr = librosa.core.load(audio_path, sr=16000)
    video_frame_list = ditto_run(SDK, audio_path, source_path)