import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image


def read_image_with_pillow(image_path):
    try:
        img = Image.open(image_path)
        img_cv = np.array(img)
        if img.mode == "RGB":
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        return img_cv
    except Exception as e:
        print(f"Can't read the picture: {image_path}, error: {e}")
        return None

# PSNR 
def calculate_psnr(image, reference):
    mse = np.mean((image.astype(np.float64) - reference.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

# UCIQE 
def calculate_uciqe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    chroma = np.sqrt(a.astype(np.float64)**2 + b.astype(np.float64)**2)
    return np.std(chroma) + np.std(l) + np.mean(l) / (np.mean(chroma) + 1e-6)

# UIQM 
def calculate_uiqm(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return 0.5 * np.std(h.astype(np.float64)) + 0.25 * np.std(s.astype(np.float64)) + 0.25 * np.mean(v.astype(np.float64))


def enhance_color_correction(img):
    b, g, r = cv2.split(img)
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
    mean_avg = (mean_r + mean_g + mean_b) / 3
    gamma_r, gamma_g, gamma_b = mean_avg / mean_r, mean_avg / mean_g, mean_avg / mean_b
    r = np.clip(r * gamma_r, 0, 255).astype(np.uint8)
    g = np.clip(g * gamma_g, 0, 255).astype(np.uint8)
    b = np.clip(b * gamma_b, 0, 255).astype(np.uint8)
    return cv2.merge((b, g, r))


def enhance_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv_enhanced = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)




def enhance_sharpness(img):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)


def enhance_image(img):
    img_color_corrected = enhance_color_correction(img)
    img_brightness_enhanced = enhance_brightness(img_color_corrected)
    img_sharp = enhance_sharpness(img_brightness_enhanced)
    return img_sharp


def enhance_and_save(image_folder, output_excel):
    results = []
    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)
        
       
        img = read_image_with_pillow(file_path)
       

        img = img.astype(np.uint8)
        reference = np.full_like(img, 128, dtype=np.uint8)

      
        psnr_before = calculate_psnr(img, reference)
        uciqe_before = calculate_uciqe(img)
        uiqm_before = calculate_uiqm(img)

     
        enhanced_img = enhance_image(img)

      
        psnr_after = calculate_psnr(enhanced_img, reference)
        uciqe_after = calculate_uciqe(enhanced_img)
        uiqm_after = calculate_uiqm(enhanced_img)

        
        enhanced_img_name = f"enhanced_{file_name}"
        try:
            cv2.imwrite(enhanced_img_name, enhanced_img)
            print(f"Save enhanced image: {enhanced_img_name}")
        except Exception as e:
            print(f"Failed to save the enhanced image: {file_name}, error: {e}")
            continue

      
        results.append({
            "image file name": file_name,
            "Degraded Image Classification": None,
            "PSNR": psnr_before,
            "UCIQE": uciqe_before,
            "UIQM": uiqm_before,
            "PSNR-IM": psnr_after,
            "UCIQE-IM": uciqe_after,
            "UIQM-IM": uiqm_after,
        })

  
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"Save enhanced image {output_excel}")


if __name__ == "__main__":
    image_folder = r"D:\Desktop\shujuji"
    output_excel = r"4.xlsx"  
    enhance_and_save(image_folder, output_excel)
