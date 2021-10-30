# import shutil
# import os
# from PIL import Image
#
# source_dir = './result'
# target_dir = '/data/haokang/Semi-Supervisied-DataSet/sketch/experiment/'
#
# if not os.path.exists(target_dir):
#     os.mkdir(target_dir)
#
# shutil.move(source_dir,os.path.join(target_dir,'result_patchnum128'))

# source_dir = '/data/haokang/Semi-Supervisied-DataSet/sketch/test'
# target_dir = '/data/haokang/Semi-Supervisied-DataSet/sketch/test_'
#
# if not os.path.exists(target_dir):
#     os.mkdir(target_dir)
#
# imgs = [os.path.join(source_dir, img) for img in os.listdir(source_dir)]
#
# for img in imgs:
#     save_name = os.path.join(target_dir, img.split('/')[-1])
#     I = Image.open(img)
#     L = I.convert('L')
#     L.save(save_name)

import os
import shutil
import random
from PIL import Image
import torchvision.transforms as transforms
from torchvision import utils

source = '/data/haokang/Semi-Supervisied-DataSet/disney_sketch/supervisied/A'
target = '/data/haokang/Semi-Supervisied-DataSet/disney_sketch/supervisied/A'

if not os.path.exists(target):
    os.mkdir(target)

imgs = [os.path.join(source, img) for img in os.listdir(source)]
random.shuffle(imgs)

transform1 = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor()
])
for i, img in enumerate(imgs):
    target_dir = os.path.join(target, str(i) + '.png')
    mode = Image.open(img)
    mode = transform1(mode)

    utils.save_image(mode, target_dir)
    print('已完成{}/{}'.format(i+1, len(imgs)))