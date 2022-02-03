import torch
from torch.nn import init
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import cv2
import numpy as np

def get_z_random(batch_size, nz, random_type='gauss'):
    if random_type == 'uni':
        z = torch.rand(batch_size, nz) * 2.0 - 1.0
    elif random_type == 'gauss':
        z = torch.randn(batch_size, nz)
    return z

def get_z_encode(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = get_z_random(std.size(0), std.size(1))
    z = eps.mul(std).add_(mu)
    return z

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

    return net

def save_model(net, name):
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
    torch.save(state_dict, name)

def cos_similarity(img, eps=1e-5):
    avg_pool = torch.nn.AvgPool2d(4)
    img_f = avg_pool(img)
    img_f = img_f.view(img_f.size(0),img_f.size(1),-1).transpose(2,1)
    img_f = img_f / (img_f.norm(dim=2)[:, :, None]+eps)
    return torch.bmm(img_f, img_f.transpose(2,1))

def edge_detection_canny(x):
    transform = transforms.Compose([transforms.ToTensor()])

    x_edge = torch.zeros(size=(x.size(0), x.size(1), x.size(2), x.size(3)))
    for i in range(x.size(0)):
        edge = transforms.ToPILImage()(x[i][0].cpu()).convert('L')
        # edge = cv2.cvtColor(np.array(edge),cv2.COLOR_RGB2BGR)
        edge = cv2.Canny(np.array(edge), 80, 250)
        # edge = Image.fromarray(cv2.cvtColor(edge,cv2.COLOR_BGR2RGB)).convert('L')
        edge = Image.fromarray(edge).convert('L')
        x_edge[i] = transform(edge).expand(1, x.size(1), x.size(2), x.size(3))

    return x_edge

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    return G.div(a * b * c * d)

def get_Gaussian_filter(noise_n):
    # Gaussian Filter
    Gaussian_filter = np.array([[1,2,1],
                                [2,4,2],
                                [1,2,1]])
    Gaussian_filter_torch = torch.from_numpy(Gaussian_filter).unsqueeze(0)
    G_filter = torch.nn.Conv2d(noise_n, noise_n, kernel_size = 3, stride = 1, padding = 1, bias=False)
    G_filter.weight.data = Gaussian_filter_torch.float().unsqueeze(0).expand(8, 8, 3, 3)

    return G_filter


def downsample(input):
    m = torch.nn.MaxPool2d(2, stride=2)
    return m(input)


def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
# def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=1.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)


def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
# def im2tensor(image, imtype=np.uint8, cent=1., factor=1.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def tensor2np(tensor_obj):
    # change dimension of a tensor object into a numpy array
    return tensor_obj[0].cpu().float().detach().numpy().transpose((1,2,0))

def np2tensor(np_obj):
     # change dimenion of np array into tensor array
    return torch.Tensor(np_obj[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def histogram(img, model):
    batch_size, channel, h, w = img.size()

    for bs in range(batch_size):
        for c in range(channel):

            img_input = img[:,c,:,:]
            img_input = img_input.unsqueeze(1)
            hist_c = model(img_input)

            if bs == 0 and c == 0:
                hist = hist_c
            else:
                hist = torch.cat([hist, hist_c], 0)

    return hist