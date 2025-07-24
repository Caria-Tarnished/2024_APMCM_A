# coding=utf-8
import cv2

def variance_of_laplacian(image):
  
    return cv2.Laplacian(image, cv2.CV_64F).var()

if __name__ == '__main__':
    
    image_path = "D:\Desktop\Attachment 1\image_067.png" 
    threshold = 300.0  # Set the fuzzy threshold

    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Can't read the picture: {image_path}")
    else:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Calculate the variance of the grayscale picture
        fm = variance_of_laplacian(gray)
        text = "Not Blurry"

        # Determine if the picture is blurry

        if fm < threshold:
            text = "Blurry"

        
        print(f"Picture path: {image_path}")
        print(f"blocked: {fm:.2f}, judge: {text}")

       # Display clarity results on the picture
        cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
