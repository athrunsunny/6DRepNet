# Import SixDRepNet
import numpy as np

from sixdrepnet import SixDRepNet
import cv2
from glob import glob
import os
from tqdm import tqdm
# Create model
# Weights are automatically downloaded
# model = SixDRepNet()
#
# img = cv2.imread(r'G:\DLproject\newflod\yolov5multask_with_norm\val_landmark_data\test_data\imgs\1_31_Waiter_Waitress_Waiter_Waitress_31_484_0.png')
#
# pitch, yaw, roll = model.predict(img)
#
# model.draw_axis(img, yaw, pitch, roll)
#
# cv2.imshow("test_window", img)
# cv2.waitKey(0)

def create_label_and_image(image_path,des_path,txt_path):
    model = SixDRepNet()

    externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
    files = list()
    for extern in externs:
        files.extend(glob(image_path + "\\*." + extern))

    for item in tqdm(files):
        img_name = '.'.join(item.replace("\\", "/").split("/")[-1].split('.')[:-1])
        img = cv2.imread(item)
        pitch, yaw, roll = model.predict(img)
        model.draw_axis(img, yaw, pitch, roll)

        save_path = os.path.join(des_path,img_name + '.jpg')
        cv2.imwrite(save_path,img)

        # save_txt_path = os.path.join(txt_path,img_name + '.txt')
        out_file = open('%s/%s.txt' % (txt_path, img_name), 'w')

        bb = np.asarray([pitch, yaw, roll]).reshape(3)
        out_file.write(" ".join([str(a) for a in bb]) + '\n')


if __name__ == '__main__':
    s_path = r'G:\deeplearning_dataset\widerface_hpfl_yolo_new\single_face_over3kb'
    d_path = r'G:\deeplearning_dataset\widerface_hpfl_yolo_new\pose_result_3k'
    os.makedirs(d_path,exist_ok=True)
    t_path = r'G:\deeplearning_dataset\widerface_hpfl_yolo_new\pose_result_txt_3k'
    os.makedirs(t_path, exist_ok=True)
    create_label_and_image(s_path,d_path,t_path)






