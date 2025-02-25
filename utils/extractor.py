import os
from PIL import Image
import torch as t
import torchvision as tv
from torch import nn
import pickle
from data.image_input import ImageDataset
from data import ImageDataLoader
import numpy as np

class Config(object):
    def __init__(self):
        return

class Extractor(object):

    def __init__(self, e_model, batch_size=128, cat_info=True,
                 vis=False, dataloader=False):
        self.batch_size = batch_size
        self.cat_info = cat_info

        self.model = e_model

        if dataloader:
            self.dataloader = dataloader
        else:
            self.transform = tv.transforms.Compose([
                tv.transforms.Resize(224),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])

    # the model's output only contains the inputs' feature
    @t.no_grad()
    def extract(self, data_root, out_root=None):
        if self.dataloader:
            return self._extract_with_dataloader(data_root=data_root, out_root=out_root)
        else:
            return self._extract_without_dataloader(data_root=data_root, cat_info=self.cat_info, out_root=out_root)

    # the model's output contains both the inputs' feature and category info
    @t.no_grad()
    def _extract_without_dataloader(self, data_root, cat_info, out_root):
        feature = []
        name = []

        self.model.eval()

        cnames = sorted(os.listdir(data_root))

        for cname in cnames:
            c_path = os.path.join(data_root, cname)
            if os.path.isdir(c_path):
                fnames = sorted(os.listdir(c_path))
                for fname in fnames:
                    path = os.path.join(c_path, fname)

                    image = Image.open(path)
                    image = self.transform(image)
                    image = image[None]
                    image = image

                    out = self.model(image)
                    if cat_info:
                        i_feature = out[1]
                    else:
                        i_feature = out

                    feature.append(i_feature.cpu().squeeze().numpy())
                    # name.append(cname + '/' + fname)
                    name = os.path.join(cname, fname)

        data = {'name': name, 'feature': feature}
        if out_root:
            out = open(out_root, 'wb')
            pickle.dump(data, out)

            out.close()
        
        return data

    # extract the inputs' feature via self.model
    # the model's output contains both the inputs' feature and category info
    # the input is loaded by dataloader
    @t.no_grad()
    def _extract_with_dataloader(self, data_root, out_root):
        names = []
        feature = []

        self.model.eval()

        opt = Config()
        opt.image_root = data_root
        opt.batch_size = 128

        # dataloader = ImageDataLoader(opt)
        # dataset = dataloader.load_data()
        dataloader = ImageDataset(data_root)

        for i, data in enumerate(dataloader):
            image = data['I']
            # image = image.unsqueeze(0)
            name = data['N']

            out = self.model(image)
            i_feature = out

            feature.append(i_feature.squeeze().numpy())

            names.append(name)

        data = {'name': names, 'feature': feature}        

        if out_root:
            out = open(out_root, 'wb')
            pickle.dump(data, out)

            out.close()

        return data

    # reload model with model object directly
    def reload_model(self, model):
        self.model = model
