import cv2
import base64
import os
from paddleocr import PaddleOCR
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_ocr_res(file_path):
    img_np = cv2.imread(file_path)
    h,w,c = img_np.shape
    img_data = {"img64": base64.b64encode(img_np).decode("utf-8"), "height": h, "width": w, "channels": c}
    result = ocr_engine(img_data)
    
    
    return result

def ocr_engine(img_data):
    ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True, show_log=False)

    img_file = img_data['img64']
    height = img_data['height']
    width = img_data['width']
    channels = img_data['channels']

    binary_data = base64.b64decode(img_file)
    img_array = np.frombuffer(binary_data, dtype=np.uint8).reshape((height, width, channels))

    if not img_file:
        return 'error: No file was uploaded.'

    result = ocr_engine.ocr(img_array)
    result = [line for line in result if line]

    ocr_result = [i[1][0] for line in result for i in line]
    return ocr_result


file_path = f'{CURRENT_DIR}/image.png'
print(get_ocr_res(file_path))
# ['确保你的Python环境配置止确，并且pip是最新版本，以避免安装过程中的问题。如果你在使用虚', '拟环境，确保你已经激活了对应的虚拟环境']