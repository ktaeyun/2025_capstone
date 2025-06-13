import numpy as np
from PIL import Image
import cv2

class UnsharpMasking:
    def __call__(self, img):
        img_np = np.array(img)
        blurred = cv2.GaussianBlur(img_np, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(img_np, 1.5, blurred, -0.5, 0)
        return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))
class CropHairOnly:
    def __call__(self, img):
        gray = img.convert("L")
        bbox = gray.point(lambda x: 255 if x > 10 else 0).getbbox()
        if bbox:
            img = img.crop(bbox)
        return img

class Gray3ch:
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        gray = img.convert("L")  # 흑백 변환
        gray_3ch = Image.merge("RGB", (gray, gray, gray))  # 3채널로 확장
        return gray_3ch

class RGBLoader:
    def __call__(self, img):
        # RGB 그대로 사용 (전처리에서 CLAHE/Gray 제거)
        return img

class CLAHEGray3ch:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, img):
        gray = np.array(img.convert("L"))
        enhanced = self.clahe.apply(gray)
        stacked = np.stack([enhanced] * 3, axis=-1)
        return Image.fromarray(stacked.astype(np.uint8))
    
    