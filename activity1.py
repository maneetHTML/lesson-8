import cv2
import matplotlib.pyplot as plt
image = cv2.imread('OIP.jpg')
image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title('RGB Image')
plt.show()
gray_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_rgb,cmap='gray')
plt.title('Gray Image')
plt.show()
cropped_image = image[100:300,200:400]
cropped_rgb=cv2.cvtColor(cropped_image,cv2.COLOR_BGR2RGB)
plt.imshow(cropped_rgb)
plt.title('cropped Image')
plt.show()