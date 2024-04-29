import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image
all_imgs = glob.glob(f"./pics/*/*.png")

margin_w = 850
margin_h = 850


for img_path in all_imgs:
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    c_h, c_w = h//2, w//2
    crop_img = img[c_h-margin_h:c_h+margin_h, c_w-margin_w:c_w+margin_w]

    Image.fromarray(crop_img).save(img_path)