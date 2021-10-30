import os
from PIL import Image
from torchvision import transforms as transforms
from torchvision import utils as vutils

dir_data = [4, 7, 9, 12, 14, 16, 18, 21, 27, 28, 29, 30, 31, 32,
            37, 47, 52, 64, 66, 68, 71, 76, 85, 95, 98, 104, 128,
            131, 138, 140, 143, 145, 146, 148, 150, 153, 160, 163,
            170, 178, 181, 184, 221, 224, 230, 238, 242, 246, 250,
            253, 260, 266, 272, 273, 282, 284]

dir_data = [str(s) for s in dir_data]

source_dir = '/data/haokang/Semi-Supervisied-DataSet/sketch/supervisied'
target_dir = '/data/haokang/Semi-Supervisied-DataSet/sketch/supervisied/clean_data/'

if not os.path.exists(target_dir):
    os.mkdir(target_dir)
    os.mkdir(os.path.join(target_dir, 'A'))
    os.mkdir(os.path.join(target_dir, 'B'))

A_imgs = [os.path.join(source_dir, 'A', img) for img in os.listdir(os.path.join(source_dir, 'A')) if img.split('.')[0] not in dir_data]
B_imgs = [os.path.join(source_dir, 'B', img) for img in os.listdir(os.path.join(source_dir, 'B')) if img.split('.')[0] not in dir_data]

transform = transforms.Compose([
        transforms.ToTensor()
])

for idx, (imgA, imgB) in enumerate(zip(A_imgs, B_imgs)):
    assert imgA.split('/')[-1] == imgB.split('/')[-1]
    imgA = Image.open(imgA)
    imgB = Image.open(imgB)
    imgA = transform(imgA)
    imgB = transform(imgB)

    nameA = os.path.join(target_dir, 'A', str(idx) + '.png')
    nameB = os.path.join(target_dir, 'B', str(idx) + '.png')
    vutils.save_image(imgA, nameA)
    vutils.save_image(imgB, nameB)
    print('已完成{} / {}'.format(idx + 1, len(A_imgs)))





