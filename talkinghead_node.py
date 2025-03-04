import folder_paths
import torchaudio
from ditto_tailinghead.stream_pipeline_offline import StreamSDK
from ditto_tailinghead.inference import ditto_run
from base_utils import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
audio_extensions = ['mp3', 'mp4', 'wav', 'ogg']


class DittoLoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in audio_extensions):
                    files.append(f)
        return {"required": {
            "audio": (sorted(files),),
            "start_time": ("FLOAT", {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
            "duration": ("FLOAT", {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
        },
        }

    CATEGORY = "TalkingHead"

    RETURN_TYPES = ("AUDIOPATH",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load_audio"

    def load_audio(self, start_time, duration, **kwargs):
        audio_file = folder_paths.get_annotated_filepath(strip_path(kwargs['audio']))
        if audio_file is None or validate_path(audio_file) != True:
            raise Exception("audio_file is not a valid path: " + audio_file)

        return ({"audio_file": audio_file, "start_time": start_time, "duration": duration},)


class DittoLoader:
    cfg_pkl = os.path.join(BASE_DIR, "models/ditto/ditto_cfg/v0.4_hubert_cfg_trt.pkl")
    data_root = os.path.join(BASE_DIR, "models/ditto/ditto_trt_Ampere_Plus")

    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                     },
                }

    CATEGORY = "TalkingHead"
    RETURN_TYPES = ("DITTOMODEL",)
    RETURN_NAMES = ("ditto-model",)

    FUNCTION = "load"

    def load(self,):
        SDK = StreamSDK(self.cfg_pkl, self.data_root)
        return (SDK, )


class DittoRunModule:
    """
    ditto运行模块
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ditto_model": ("DITTOMODEL", ),
                "image": ("IMAGE", ),
                "audio": ("AUDIO", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video",)

    CATEGORY = "TalkingHead"
    FUNCTION = "run"

    OUTPUT_NODE = True

    def run(self, ditto_model, image, audio):
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]

        # 目标采样率为 16000 Hz
        target_sample_rate = 16000
        # 创建重采样器
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate
        )
        # 对音频波形进行重采样
        resampled_waveform = resampler(waveform)
        # 将重采样后的 torch.Tensor 转换为 numpy 数组
        resampled_numpy = resampled_waveform.numpy()

        # 如果音频是单声道，可以去掉通道维度
        if resampled_numpy.shape[0] == 1:
            resampled_numpy = resampled_numpy.squeeze(0)
        else:  # 多声道音频，可以选择合并或处理每个通道
            resampled_numpy = waveform.mean(dim=0).numpy()  # 简单地取平均值合并通道

        image = tensor2cvdata(image, mode="rgb")
        video_frame_list = ditto_run(ditto_model, resampled_numpy, [image])
        video_frame_list = [cv2tensor(i, mode="bgr")for i in video_frame_list]
        video_frame = torch.cat(video_frame_list, dim=0)
        return (video_frame, )