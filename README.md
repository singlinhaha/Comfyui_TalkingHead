# Comfyui_TalkingHead

本仓库是对ditto-talkinghead的非官方comfyui实现。
**原仓库: [ditto-talkinghead](https://github.com/antgroup/ditto-talkinghead)，感谢作者的分享**
**功能：**

* 上传音频生成数字人视频

**注:**

1. comfyui限定了numpy版本为1.26.4, 原仓库需要的版本为2.0.1

## 环境安装

#### 环境依赖：

```bash
pip install -r requirements.txt
```

#### 下载模型
模型地址：[HuggingFace](https://huggingface.co/digital-avatar/ditto-talkinghead/tree/main)
模型下载后放在：custom_nodes/Comfyui_TalkingHead/models/ditto
```bash
./models/ditto/
├── ditto_cfg
│   ├── v0.4_hubert_cfg_trt.pkl
│   └── v0.4_hubert_cfg_trt_online.pkl
├── ditto_onnx
│   ├── appearance_extractor.onnx
│   ├── blaze_face.onnx
│   ├── decoder.onnx
│   ├── face_mesh.onnx
│   ├── hubert.onnx
│   ├── insightface_det.onnx
│   ├── landmark106.onnx
│   ├── landmark203.onnx
│   ├── libgrid_sample_3d_plugin.so
│   ├── lmdm_v0.4_hubert.onnx
│   ├── motion_extractor.onnx
│   ├── stitch_network.onnx
│   └── warp_network.onnx
└── ditto_trt_Ampere_Plus
    ├── appearance_extractor_fp16.engine
    ├── blaze_face_fp16.engine
    ├── decoder_fp16.engine
    ├── face_mesh_fp16.engine
    ├── hubert_fp32.engine
    ├── insightface_det_fp16.engine
    ├── landmark106_fp16.engine
    ├── landmark203_fp16.engine
    ├── lmdm_v0.4_hubert_fp32.engine
    ├── motion_extractor_fp32.engine
    ├── stitch_network_fp16.engine
    └── warp_network_fp16.engine
```
