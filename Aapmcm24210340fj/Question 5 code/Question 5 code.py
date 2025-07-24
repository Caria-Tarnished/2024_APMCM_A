import pandas as pd
import matplotlib.pyplot as plt


output_excel = "D:\Desktop\Answer_5.xlsx"
data = pd.read_excel(output_excel)

# Visual PSNR comparison
plt.figure(figsize=(10, 6))
plt.plot(data["image file name"], data["PSNR (Original)"], marker='o', label="PSNR (Original)")
plt.plot(data["image file name"], data["PSNR (Comprehensive)"], marker='o', label="PSNR (Comprehensive)", color='orange')
plt.title("PSNR Comparison: Original vs Comprehensive Enhancement")
plt.xlabel("Image File Name")
plt.ylabel("PSNR")
plt.xticks([])  
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Visual UCIQE comparison
plt.figure(figsize=(10, 6))
plt.plot(data["image file name"], data["UCIQE (Original)"], marker='o', label="UCIQE (Original)")
plt.plot(data["image file name"], data["UCIQE (Comprehensive)"], marker='o', label="UCIQE (Comprehensive)", color='orange')
plt.title("UCIQE Comparison: Original vs Comprehensive Enhancement")
plt.xlabel("Image File Name")
plt.ylabel("UCIQE")
plt.xticks([])  
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Visual UIQM comparison
plt.figure(figsize=(10, 6))
plt.plot(data["image file name"], data["UIQM (Original)"], marker='o', label="UIQM (Original)")
plt.plot(data["image file name"], data["UIQM (Comprehensive)"], marker='o', label="UIQM (Comprehensive)", color='orange')
plt.title("UIQM Comparison: Original vs Comprehensive Enhancement")
plt.xlabel("Image File Name")
plt.ylabel("UIQM")
plt.xticks([])  
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
