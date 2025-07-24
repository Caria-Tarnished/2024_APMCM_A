import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

image_dir = "D:\Desktop\Attachment 1"  
images = os.listdir(image_dir)


channel_differences = []  
brightness_values = []    
laplacian_values = []     

for img_name in images:
   
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)
    
   
    if img is None:
        print(f"Failed to load image: {img_name}")
        continue

   
    mean_r, mean_g, mean_b = np.mean(img[:, :, 2]), np.mean(img[:, :, 1]), np.mean(img[:, :, 0])
    diff_rg = abs(mean_r - mean_g) / max(mean_r, mean_g, 1e-5)
    diff_gb = abs(mean_g - mean_b) / max(mean_g, mean_b, 1e-5)
    diff_rb = abs(mean_r - mean_b) / max(mean_r, mean_b, 1e-5)
    channel_differences.append(max(diff_rg, diff_gb, diff_rb))  
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    mean_brightness = np.mean(gray)
    brightness_values.append(mean_brightness)  

   
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  
    laplacian_values.append(laplacian_var) 


def plot_distribution(data, title, xlabel, ylabel="Frequency", bins=30):
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=bins, color="blue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


plot_distribution(channel_differences, 
                  title="Color Difference Distribution", 
                  xlabel="Max Channel Difference Ratio")


plot_distribution(brightness_values, 
                  title="Brightness Distribution", 
                  xlabel="Average Brightness")


plot_distribution(laplacian_values, 
                  title="Laplacian Variance Distribution", 
                  xlabel="Laplacian Variance")
