import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image, ImageFilter
import glob
import os
import numpy as np

def getData(mode, root=None):
    if mode =='train_IHN':
        colorized_path = r'' + os.path.join(root, 'helen/helen_[1-4]/*.jpg')
        colorized_paths = glob.glob(colorized_path)
        halftone_path = r'' + os.path.join(root, 'helen_h_256/helen_[1-4]/*.jpg')
        halftone_paths = glob.glob(halftone_path)
        
        return halftone_paths, colorized_paths, None

    elif mode =='train':
        # colorized_paths = glob.glob(r'./dataset/helen/helen_[1-4]/*.jpg')
        # halftone_paths = glob.glob(r'./dataset/helen_h_256/helen_[1-4]/*.jpg')
        # c_halftone_paths = glob.glob(r'./dataset/helen_ch_256/helen_[1-4]/*.jpg')
        halftone_paths = glob.glob(r'/eva_data_0/katherine/Halftone/Dataset/helen_h_256/helen_[1-4]/*.jpg')
        colorized_paths = glob.glob(r'/eva_data_0/katherine/Halftone/helen/helen_[1-4]/*.jpg')
        c_halftone_paths = glob.glob(r'/eva_data_0/katherine/Halftone/Dataset/helen_ch_256/helen_[1-4]/*.jpg')
        
        return halftone_paths, colorized_paths, c_halftone_paths

    else:
        filenames = os.listdir(root)

        return filenames, None, None

class IMGLoader(data.Dataset):
    def __init__(self, mode, size, root, ref_root = './samples/ref_image/'):
        self.mode = mode
        self.size = size
        self.root = root
        self.ref_root = ref_root
        self.bw_halftone_paths, self.colorized_paths, self.c_halftone_paths = getData(mode, root)
        self.transform = transforms.Compose(
                            [transforms.Resize((size), interpolation=Image.BILINEAR),
                             transforms.CenterCrop(size),
                             transforms.ToTensor()])
        if mode == 'train':
            self.reference_list_imagenet = np.load('/eva_data_1/yuting/our/code/reference_list_imagenet.npy')
            self.reference_list_imagenet[379][0] = self.reference_list_imagenet[379][1]
            self.reference_list_imagenet[242][1] = self.reference_list_imagenet[242][0]
            self.reference_list_imagenet[1451][2] = self.reference_list_imagenet[1451][0]
            self.reference_list_imagenet[1807][2] = self.reference_list_imagenet[1807][0]

            self.reference_list = np.load('./reference_list.npy')
            self.reference_list[187][4] = self.reference_list[187][3]
            self.reference_list[188][0] = self.reference_list[188][1]
            self.reference_list[529][3] = self.reference_list[529][2]
            self.reference_list[682][3] = self.reference_list[682][2]
        else:
            self.reference_list = np.load('/eva_data_1/yuting/our/code/test_ref_list.npy')
            # print(self.colorized_paths[281])
            # print(self.colorized_paths[self.reference_list[281][0]])
            self.reference_list[281][0] = self.reference_list[281][1]

            self.reference_list[70][1] = self.reference_list[70][0]

            # real test warp 
            # self.reference_list[62][1] = self.reference_list[62][2]

        self.length = len(self.bw_halftone_paths)

    def __len__(self):
        return len(self.bw_halftone_paths)

    def __getitem__(self, index):
        if self.mode == 'train_IHN':
            # halftone image
            halftone_path = self.bw_halftone_paths[index]
            halftone_img = Image.open(halftone_path).convert('L')
            halftone_img = self.transform(halftone_img)
            halftone_img = halftone_img.type(torch.cuda.FloatTensor)

            # grayscale ground truth
            colorized_path = self.colorized_paths[index]
            bw_img = Image.open(colorized_path).convert('L')
            bw_img = self.transform(bw_img)
            bw_img = bw_img.type(torch.cuda.FloatTensor)

            return halftone_img, bw_img

        elif self.mode == 'train':
            colorized_path = self.colorized_paths[index]
            halftone_path = self.bw_halftone_paths[index]
            c_halftone_path = self.c_halftone_paths[index]

            colorized_img = Image.open(colorized_path).convert('RGB')
            bw_img = Image.open(colorized_path).convert('L')
            halftone_img = Image.open(halftone_path).convert('L')
            c_halftone_img = Image.open(c_halftone_path).convert('RGB')

            colorized_img = self.transform(colorized_img)
            colorized_img = colorized_img.type(torch.cuda.FloatTensor)
            bw_img = self.transform(bw_img)
            bw_img = bw_img.type(torch.cuda.FloatTensor)
            halftone_img = self.transform(halftone_img)
            halftone_img = halftone_img.type(torch.cuda.FloatTensor)
            c_halftone_img = self.transform(c_halftone_img)
            c_halftone_img = c_halftone_img.type(torch.cuda.FloatTensor)



            prop = np.random.uniform(0, 1, 1)
            if  index >= 1500:
                tag = 1
            elif prop > 1/3:
                tag = 0
            else:
                tag = 1

            if tag == 0:
                rand_idx = np.random.randint(0, 5)

                # if rand_idx == 5:
                #     return colorized_img, halftone_img, bw_img, colorized_img, colorized_img

                ref_idx = self.reference_list[index][rand_idx]

                ref_colorized_path = self.colorized_paths[ref_idx]
                ref_colorized_img = Image.open(ref_colorized_path).convert('RGB')
                ref_colorized_img = self.transform(ref_colorized_img)
                ref_colorized_img = ref_colorized_img.type(torch.cuda.FloatTensor)

                warp_path = '/eva_data_1/yuting/other/Deep-Exemplar-based-Colorization/demo/train/res/'+ \
                            colorized_path.split('/')[-1][:-4] + '_gray_' + ref_colorized_path.split('/')[-1][:-4] + '.png'

            else:
                rand_idx = np.random.randint(0, 3)

                # if rand_idx == 5:
                #     return colorized_img, halftone_img, bw_img, colorized_img, colorized_img

                ref_name = self.reference_list_imagenet[index][rand_idx]

                ref_colorized_path = '/eva_data_1/yuting/other/imagenet/image6000/'+ ref_name
                ref_colorized_img = Image.open(ref_colorized_path).convert('RGB')
                ref_colorized_img = self.transform(ref_colorized_img)
                ref_colorized_img = ref_colorized_img.type(torch.cuda.FloatTensor)

                warp_path = '/eva_data_1/yuting/other/Deep-Exemplar-based-Colorization/demo/train_imagenet/res/'+ \
                            colorized_path.split('/')[-1][:-4] + '_gray_' + ref_name[:-4] + '.png'
            # warp_path = ref_colorized_path
            warp_colorized_img = Image.open(warp_path).convert('RGB')
            warp_colorized_img = self.transform(warp_colorized_img)
            warp_colorized_img = warp_colorized_img.type(torch.cuda.FloatTensor)

            return colorized_img, halftone_img, c_halftone_img, bw_img, ref_colorized_img, warp_colorized_img

        else:
            halftone_path = self.root + 'input_%d.jpg' % index
            halftone_img = Image.open(halftone_path).convert('L')
            halftone_img = self.transform(halftone_img)
            halftone_img = halftone_img.type(torch.cuda.FloatTensor)

            ref_colorized_path = self.ref_root + 'ref_%d.jpg' % index
            ref_colorized_img = Image.open(ref_colorized_path).convert('RGB')
            ref_colorized_img = self.transform(ref_colorized_img)
            ref_colorized_img = ref_colorized_img.type(torch.cuda.FloatTensor)

            return halftone_img, ref_colorized_img