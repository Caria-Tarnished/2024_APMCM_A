import cv2
import os
import math
import pandas as pd
from imutils import paths
import numpy as np

# PSNR
def psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:  
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# UCIQE
def uciqe(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = np.std(chroma)
    mu_c = np.mean(chroma)
    mu_s = np.mean(l)
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return c1 * sigma_c + c2 * mu_c + c3 * mu_s

# UIQM
def uiqm(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    laplacian_var = cv2.Laplacian(l_channel, cv2.CV_64F).var()
    contrast = np.max(l_channel) - np.min(l_channel)
    color_mean = np.mean(a_channel) + np.mean(b_channel)
    return 0.5 * laplacian_var + 0.3 * contrast + 0.2 * color_mean


def is_low_light(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r, c = gray_img.shape[:2]
    dark_sum = sum(sum(1 for colum in row if colum < 40) for row in gray_img)
    dark_prop = dark_sum / (r * c)
    return dark_prop >= 0.6


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

    return result > 3.5


def is_blurry(img, threshold=250):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold


def classify_and_save_to_excel(folder_path, excel_path):
    results = []
    for image_path in paths.list_images(folder_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Can't read the picture: {image_path}")
            continue

   
        classification = "Normal"
        if is_low_light(image):
            classification = "low light"
        elif has_color_bias(image):
            classification = "color cast"
        elif is_blurry(image):
            classification = "blur"

        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        psnr_value = psnr(gray_image, gray_image)  
        uciqe_value = uciqe(image)
        uiqm_value = uiqm(image)

      
        results.append({
            "image file name": os.path.basename(image_path),
            "Degraded Image Classification": classification,
            "PSNR": psnr_value,
            "UCIQE": uciqe_value,
            "UIQM": uiqm_value
        })


    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False)
    print(f"Classification results have been saved to {excel_path}")


if __name__ == "__main__":
    folder_path = "D:\Desktop\shujuji" 
    excel_path = "image_classification_results.xlsx"  
    classify_and_save_to_excel(folder_path, excel_path)
