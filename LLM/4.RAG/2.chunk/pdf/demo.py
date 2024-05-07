
import os
import fitz
from tqdm import tqdm
import numpy as np
import base64
from paddleocr import PaddleOCR

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pdf_ocr_txt(filepath, dir_path="tmp_files"):
    full_dir_path = os.path.join(os.path.dirname(filepath), dir_path)
    if not os.path.exists(full_dir_path):
        os.makedirs(full_dir_path)
    doc = fitz.open(filepath)
    txt_file_path = os.path.join(full_dir_path, "{}.txt".format(os.path.split(filepath)[-1]))
    img_name = os.path.join(full_dir_path, 'tmp.png')
    with open(txt_file_path, 'w', encoding='utf-8') as fout:
        for i in tqdm(range(doc.page_count)):
            page = doc.load_page(i)
            pix = page.get_pixmap() # 将 PDF 页面转换成一个图像
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))

            img_data = {"img64": base64.b64encode(img).decode("utf-8"), "height": pix.h, "width": pix.w,
                        "channels": pix.n}
            result = ocr_ocr(img_data)
            result = [line for line in result if line]
            ocr_result = [i[1][0] for line in result for i in line]
            fout.write("\n".join(ocr_result))
    if os.path.exists(img_name):
        os.remove(img_name)
    return txt_file_path



def ocr_ocr(img_data):
    # 初始化 PaddleOCR 引擎
    ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True, show_log=False)


    img_file = img_data['img64']
    height = img_data['height']
    width = img_data['width']
    channels = img_data['channels']

    binary_data = base64.b64decode(img_file)
    img_array = np.frombuffer(binary_data, dtype=np.uint8).reshape((height, width, channels))

    # 无文件上传，返回错误
    if not img_file:
        return 'error: No file was uploaded.'

    # 调用 PaddleOCR 进行识别
    res = ocr_engine.ocr(img_array)

    # 返回识别结果
    return res



if __name__ == "__main__":
    # from torch.backends import cudnn
    # print(cudnn.is_available())
    #
    pdf_path = f'{CURRENT_DIR}/data/image-based.pdf'
    res = pdf_ocr_txt(pdf_path)
    print(res)
    # 
    # print(os.getenv("OCR_USE_GPU")) # None
