import cv2
import os
import math
from imutils import paths


low_light_dir = "LowLightImages"
color_bias_dir = "ColorBiasImages"
blurry_dir = "BlurryImages"
normal_dir = "NormalImages"


for dir_path in [low_light_dir, color_bias_dir, blurry_dir, normal_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def is_low_light(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r, c = gray_img.shape[:2]
    dark_sum = sum(sum(1 for colum in row if colum < 40) for row in gray_img)
    dark_prop = dark_sum / (r * c)
    return dark_prop >= 0.3


def has_color_bias(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    h, w = img.shape[:2]
    da = a_channel.sum() / (h * w) - 128
    db = b_channel.sum() / (h * w) - 128

    histA = [0] * 256
    histB = [0] * 256
    for i in range(h):
        for j in range(w):
            histA[a_channel[i][j]] += 1
            histB[b_channel[i][j]] += 1

    msqA = sum(float(abs(y - 128 - da)) * histA[y] / (w * h) for y in range(256))
    msqB = sum(float(abs(y - 128 - db)) * histB[y] / (w * h) for y in range(256))
    result = math.sqrt(da**2 + db**2) / math.sqrt(msqA**2 + msqB**2)

    return result > 3


def is_blurry(img, threshold=300):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold


def classify_images(folder_path):
    for image_path in paths.list_images(folder_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Can't read the picture: {image_path}")
            continue

        if is_low_light(image):
            save_path = os.path.join(low_light_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, image)
            print(f" {image_path} Classified as LowLightImages")
        elif has_color_bias(image):
            save_path = os.path.join(color_bias_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, image)
            print(f" {image_path} Classified as ColorBiasImages")
        elif is_blurry(image):
            save_path = os.path.join(blurry_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, image)
            print(f" {image_path} Classify as BlurryImages")
        else:
            save_path = os.path.join(normal_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, image)
            print(f" {image_path} Classified as ColorBiasImages")


if __name__ == "__main__":
    folder_path ="D:\Desktop\shujuji"  
    classify_images(folder_path)
