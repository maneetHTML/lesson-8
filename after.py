import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('OIP.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(image_rgb)
plt.title('RGB Image')
plt.axis('off')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

cropped = image[100:300, 200:400]
cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(cropped_rgb)
plt.title('Cropped Image')
plt.axis('off')

(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)  
rotated = cv2.warpAffine(image, M, (w, h))
rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(rotated_rgb)
plt.title('Rotated Image (45°)')
plt.axis('off')


brightness_matrix = np.ones_like(image, dtype='uint8') * 50
brighter = cv2.add(image, brightness_matrix)
brighter_rgb = cv2.cvtColor(brighter, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(brighter_rgb)
plt.title('Brighter Image (+50)')
plt.axis('off')

plt.show()
