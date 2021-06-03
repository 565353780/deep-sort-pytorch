import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import os

from .model import Net, osnet_ibn_x1_0

USE_OSNet = True
OSNet_Path = 'models/osnet_ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth'

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        if USE_OSNet:
            self.net = osnet_ibn_x1_0(pretrained=False)
            self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
            state_dict = torch.load(OSNet_Path)
            self.net.load_state_dict(state_dict)
            self.net.eval()
            print("Loading weights from {}... Done!".format(OSNet_Path))
            self.net.to(self.device)
            self.size = (256, 128)
            self.norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.net = Net(reid=True)
            self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
            self.net.load_state_dict(state_dict)
            print("Loading weights from {}... Done!".format(model_path))
            self.net.to(self.device)
            self.size = (64, 128)
            self.norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        


    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

