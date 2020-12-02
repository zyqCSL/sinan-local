import io
import base64
from PIL import Image
import sys
import os

image_dir = './base64_images/'
media_jpg_num = 17
media_png_num = 15

jpg_b64 = {}
png_b64 = {}

for i in range(1, media_jpg_num + 1):
	with open(image_dir + str(i) + '.jpg', 'rb') as f:
		jpg_b64[i] = str(f.read())

for i in range(1, media_png_num + 1):
	with open(image_dir + str(i) + '.png', 'rb') as f:
		png_b64[i] = str(f.read())

for i in range(1, media_jpg_num + 1):
	jpg_img = base64.b64decode(jpg_b64[i])
	tempBuff = io.BytesIO()
	tempBuff.write(jpg_img)
	tempBuff.flush()
	image = Image.open(tempBuff)
	tempBuff.close()      
	print(type(image))
	print(image.format)
	print(image.mode)
	print(image.size)

for i in range(1, media_png_num + 1):
	png_img = base64.b64decode(png_b64[i])
	tempBuff = io.BytesIO()
	tempBuff.write(png_img)
	tempBuff.flush()
	image = Image.open(tempBuff)
	tempBuff.close()      
	print(type(image))
	print(image.format)
	print(image.mode)
	print(image.size)
