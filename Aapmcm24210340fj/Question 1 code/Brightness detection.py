import cv2
import os

def analyze_image(img, pic_path):
  
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gets the number of rows and columns of the grayscale matrix
    r, c = gray_img.shape[:2]
    dark_sum = 0  
    piexs_sum = r * c  

   
    for row in gray_img:
        for colum in row:
            if colum < 40:  
                dark_sum += 1
    dark_prop = dark_sum / piexs_sum  
    print(f"Detection picture: {pic_path}")
    print(f"dark_sum: {dark_sum}")
    print(f"piexs_sum: {piexs_sum}")
    print(f"dark_prop: {dark_prop:.2f}")

   
    if dark_prop >= 0.3:  
        print(f"{pic_path} is dark!")
        save_dark_image(img, pic_path)  
    else:
        print(f"{pic_path} is bright!")

def save_dark_image(img, pic_path):
 
    dark_dir = "../DarkPicDir"
    
    if not os.path.exists(dark_dir):
        os.makedirs(dark_dir)
    
   
    pic_name = os.path.basename(pic_path)
    save_path = os.path.join(dark_dir, pic_name)
    
    cv2.imwrite(save_path, img)
    print(f"The picture was saved to: {save_path}")

if __name__ == "__main__":
   
    image_path = "D:\\Desktop\\shujuji\\78_img_.png"
    
   
    if not os.path.exists(image_path):
        print(f"Invalid picture path: {image_path}")
    else:
       
        img = cv2.imread(image_path)
        if img is None:
            print(f"Can't read the picture: {image_path}")
        else:
            analyze_image(img, image_path)
