import cv2
import os
import hashlib
import numpy as np
import torch


def tensor2cvdata(image, mode="bgr"):
    """
    将tensor数据转为cv2数据
    Args:
        image: np.array
    Returns:

    """
    image = image.squeeze(0) if image.dim() == 4 else image
    image = (image.cpu().numpy() * 255).astype(np.uint8)
    if mode == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def cv2tensor(image, mode="bgr"):
    """
    将cv2数据转为tensor数据
    Args:
        image: tensor
    Returns:
    """
    if mode == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image


def is_url(url):
    return url.split("://")[0] in ["http", "https"]


# modified from https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
def calculate_file_hash(filename: str, hash_every_n: int = 1):
    #Larger video files were taking >.5 seconds to hash even when cached,
    #so instead the modified time from the filesystem is used as a hash
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()


def strip_path(path):
    #This leaves whitespace inside quotes and only a single "
    #thus ' ""test"' -> '"test'
    #consider path.strip(string.whitespace+"\"")
    #or weightier re.fullmatch("[\\s\"]*(.+?)[\\s\"]*", path).group(1)
    path = path.strip()
    if path.startswith("\""):
        path = path[1:]
    if path.endswith("\""):
        path = path[:-1]
    return path

def hash_path(path):
    if path is None:
        return "input"
    if is_url(path):
        return "url"
    return calculate_file_hash(strip_path(path))


def is_safe_path(path):
    if "VHS_STRICT_PATHS" not in os.environ:
        return True
    basedir = os.path.abspath('.')
    try:
        common_path = os.path.commonpath([basedir, path])
    except:
        #Different drive on windows
        return False
    return common_path == basedir


def validate_path(path, allow_none=False, allow_url=True):
    if path is None:
        return allow_none
    if is_url(path):
        #Probably not feasible to check if url resolves here
        if not allow_url:
            return "URLs are unsupported for this path"
        return is_safe_path(path)
    if not os.path.isfile(strip_path(path)):
        return "Invalid file path: {}".format(path)
    return is_safe_path(path)
