import os, sys
main_path = os.path.dirname(__file__)
sys.path.append(main_path)

from talkinghead_node import (DittoLoader, DittoRunModule, DittoLoadAudio)

NODE_CLASS_MAPPINGS = {
    "DittoLoader": DittoLoader,
    "DittoRunModule": DittoRunModule,
    "DittoLoadAudio": DittoLoadAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DittoLoader": "Ditto加载器",
    "DittoRunModule": "Ditto运行模块",
    "DittoLoadAudio": "🎤 音频选择",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']