from utils.extractor import Extractor
from models.vgg import vgg16
# from models.sketch_resnet import resnet50
import torch as t
from torch import nn
import os

# The script to extract sketches or photos' features using the trained model
train_set_root = 'dataset/sketch_train'
test_set_root = 'dataset/sketch_test'

train_photo_root = 'dataset/photo_train'
test_photo_root = 'dataset/photo_test'

# The trained model root for vgg
PHOTO_VGG = 'model/photo_vgg16_29.pth'
a = PHOTO_VGG.split('_')
epoch =''+ a[-1].split('.')[0]

device = 'cuda:1'

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''vgg'''
vgg = vgg16(pretrained=False)
vgg.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)
vgg.load_state_dict(t.load(PHOTO_VGG, map_location=t.device('cpu')), strict=False)

ext = Extractor(vgg)
ext.reload_model(vgg)
vgg.eval()

photo_feature = ext._extract_with_dataloader(test_photo_root, 'feature/bt32_1e3_1/photo-vgg' + '-%sepoch.pkl'%epoch)

sketch_feature = ext._extract_with_dataloader(test_set_root, 'feature/bt32_1e3_1/sketch-vgg' + '-%sepoch.pkl'%epoch)
