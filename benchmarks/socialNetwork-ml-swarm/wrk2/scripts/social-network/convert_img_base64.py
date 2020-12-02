import sys
import os
import base64

image_dir = './images/'
target_dir = './base64_images/'
media_jpg_num = 17
media_png_num = 15

for i in range(1, media_jpg_num + 1):
  with open(image_dir + str(i) + '.jpg', 'rb') as f:
    b64_img = str(base64.b64encode(f.read())).decode('utf-8')
    with open(target_dir + str(i) + '.jpg', 'w+') as f:
      f.write(b64_img)

for i in range(1, media_png_num + 1):
  with open(image_dir + str(i) + '.png', 'rb') as f:
    b64_img = str(base64.b64encode(f.read())).decode('utf-8')
    with open(target_dir + str(i) + '.png', 'w+') as f:
      f.write(b64_img)