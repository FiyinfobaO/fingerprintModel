import numpy as np
import argparse
from PIL import Image
import cv2

img = cv2.imread('fake.BMP')
cv2.imwrite('1.BMP',img)
print(img.shape)
print(img.size)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
cv2.imwrite('2.BMP',img)
input_shape = [1, 96, 96, 3]
size = input_shape[1:3]
img.resize(size)
cv2.imwrite('3.BMP',img)
print(img.shape)
print(img.size)
img = np.array(img, dtype=np.float32)
cv2.imwrite('4.BMP',img)
print(img.shape)
print(img.size)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
cv2.imwrite('5.BMP',img)
print(img.shape)
input_data = np.expand_dims(img, axis=0)
print(input_data.shape)
print(input_data.size)

img = Image.open('fake.BMP').convert('RGB')
cv2.imwrite('1a.BMP',img)
print(img.size)
input_shape = [1, 96, 96, 3]
size = input_shape[1:3]
img = img.resize(size)
cv2.imwrite('2a.BMP',img)
print(img.size)
img = np.array(img, dtype=np.float32)
cv2.imwrite('3a.BMP',img)
print(img.size)

input_data = np.expand_dims(img, axis=0)
cv2.imwrite('4a.BMP',input_data)
print(input_data.size)
# img = cv2.cvtColor(img, cv2.C)
# print(img.shape)
# print(img.size)