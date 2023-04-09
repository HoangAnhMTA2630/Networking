import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import numpy as np
import shutil

model = tf.keras.models.load_model('models/imageclassifier.h5')

image_folder_path = "test_data"

# Đường dẫn đến thư mục lưu trữ ảnh đã được kiểm tra và chuyển sang thư mục này nếu ảnh đáp ứng điều kiện
output_folder_1 = "./output/baodong/"

# Đường dẫn đến thư mục lưu trữ ảnh không đáp ứng điều kiện và chuyển sang thư mục này nếu ảnh không đáp ứng điều kiện
output_folder_2 = "./output/boinhocanhan/"

image_files = os.listdir(image_folder_path)
length = len(image_files)

for i in image_files:
    file_path = os.path.join("./test_data/", i)
    img = cv2.imread(file_path)
    resize = tf.image.resize(img, (256,256))
    yhat = model.predict(np.expand_dims(resize/255, 0))
    if yhat > 0.5: 
        print(f'bôi nhọ cá nhân')
        shutil.copy2(file_path, output_folder_2)
    else:
        print(f'bạo động')
        shutil.copy2(file_path, output_folder_1)