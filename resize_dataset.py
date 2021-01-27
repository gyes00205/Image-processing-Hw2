import cv2
import glob
import os

cat_dir = 'train/cat/'
dog_dir = 'train/dog/'
cat_filenames = os.listdir(cat_dir)
dog_filenames = os.listdir(dog_dir)
print(cat_filenames)
print(dog_filenames)

for filename in cat_filenames:
    img = cv2.imread(cat_dir + filename)
    img1 = cv2.resize(img.copy(), (400,400))
    img2 = cv2.resize(img.copy(), (200,400))
    img3 = cv2.resize(img.copy(), (400,200))
    cv2.imwrite(cat_dir+filename[:-4]+'_r1.jpg', img1)
    cv2.imwrite(cat_dir+filename[:-4]+'_r2.jpg', img2)
    cv2.imwrite(cat_dir+filename[:-4]+'_r3.jpg', img3)

for filename in dog_filenames:
    img = cv2.imread(dog_dir + filename)
    img1 = cv2.resize(img.copy(), (400,400))
    img2 = cv2.resize(img.copy(), (200,400))
    img3 = cv2.resize(img.copy(), (400,200))
    cv2.imwrite(dog_dir+filename[:-4]+'_r1.jpg', img1)
    cv2.imwrite(dog_dir+filename[:-4]+'_r2.jpg', img2)
    cv2.imwrite(dog_dir+filename[:-4]+'_r3.jpg', img3)