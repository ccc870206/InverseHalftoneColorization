import numpy as np
from torch.utils.data import DataLoader
import torch
import argparse
from models.model_G_F import *
from models.model_IHN import IHNet
from dataloader import *
from functions import *
from torchvision.utils import save_image

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weight_dir',type=str,help='directory of model weights',default='./weights/')
    parser.add_argument('--input_dir',type=str,help='directory of input images',default='./samples/input_image/')
    parser.add_argument('--ref_dir',type=str,help='directory of reference images',default='./samples/ref_image/')
    parser.add_argument('--result_save_dir',type=str,help='directory of results',default='./results/')
    parser.add_argument('--batch_size',type=int,help='batchsize',default=1)
    parser.add_argument('--noise_n',type=int,help='length of noise',default=8)
    parser.add_argument('--img_size',type=int,help='size of image',default=256)
    parser.add_argument('--gpu_ids', type=str, default='0',
                                help='gpu ids: e.g. 0, 1, 2 use -1 for CPU')
                                
    args = parser.parse_args()
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >=0:
            args.gpu_ids.append(id)
    return args

args = get_args()

os.makedirs(args.result_save_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
GPUID = args.gpu_ids
torch.cuda.set_device(GPUID[0])

# MODEL
IHN = IHNet()
E = Encoder(3,3)
G = G_Unet_add_input()
F = MultipleBasicBlock_4(6)

if args.noise_n > 0:
    G_filter = get_Gaussian_filter(args.noise_n)
    G_filter.cuda(GPUID[0])

IHN.load_state_dict(torch.load('%sIHN.pth' % args.weight_dir))
E.load_state_dict(torch.load('%sE.pth' % args.weight_dir))
G.load_state_dict(torch.load('%sG.pth' % args.weight_dir))
F.load_state_dict(torch.load('%sF.pth' % args.weight_dir))

IHN.cuda(GPUID[0]).eval()
E.cuda(GPUID[0]).eval()
G.cuda(GPUID[0]).eval()
F.cuda(GPUID[0]).eval()

# DATA
dataset = IMGLoader('test', args.img_size, root = args.input_dir, ref_root = args.ref_dir)
loader = data.DataLoader(dataset, batch_size = args.batch_size,
                                  shuffle = False,
                                  drop_last = False)

with torch.no_grad():
    for step, (h_img, ref_img) in enumerate(loader):
        # Encode
        z_mu_ref, z_logvar_ref = E(ref_img.cuda(GPUID[0]))
        z_ref = get_z_encode(z_mu_ref.cpu(), z_logvar_ref.cpu())

        z_random_1 = get_z_random(h_img.size(0), 8)
        z_random_2 = get_z_random(h_img.size(0), 8)

        # ----------
        #   IHN->G
        # ----------
        # Halftone
        if args.noise_n >0:
            noise = torch.randn(h_img.size(0), args.noise_n, args.img_size, args.img_size).cuda(GPUID[0])
            noise = G_filter(noise)

        bw_ph_ihn = IHN(h_img.cuda(GPUID[0]), z=noise.cuda(GPUID[0]))[-1]

        # Edge detection canny
        bw_ph_edge_ihn = edge_detection_canny(bw_ph_ihn)

        # Generate colorized img
        c_img_forward_ref = G(bw_ph_ihn.cuda(GPUID[0]), bw_ph_edge_ihn.cuda(GPUID[0]), z_ref.cuda(GPUID[0]))

        # ----------
        #   G->IHN
        # ----------
        # Generate color halftone
        bw_ht_edge = edge_detection_canny(h_img)
        c_ht_ref = G(h_img.cuda(GPUID[0]), bw_ht_edge.cuda(GPUID[0]), z_ref.cuda(GPUID[0]))

        # Generate colorized img
        for channel in range(3):
            if channel == 0:
                c_img_inv_ref = IHN(c_ht_ref[:, channel, :, :].cuda(GPUID[0]).unsqueeze(1), z=noise.cuda(GPUID[0]))[-1]
            else:            
                c_img_inv_ref = torch.cat([c_img_inv_ref, IHN(c_ht_ref[:, channel, :, :].cuda(GPUID[0]).unsqueeze(1), z=noise.cuda(GPUID[0]))[-1]], 1)
    
        # ----------
        #   Fuse
        # ----------
        c_img_ref = F(c_img_forward_ref.cuda(GPUID[0]), c_img_inv_ref.cuda(GPUID[0]))

        # ----------
        # Random n
        # ----------
        # Generate random colorized img
        c_img_forward_rand_1 = G(bw_ph_ihn.cuda(GPUID[0]), bw_ph_edge_ihn.cuda(GPUID[0]), z_random_1.cuda(GPUID[0]))
        c_img_forward_rand_2 = G(bw_ph_ihn.cuda(GPUID[0]), bw_ph_edge_ihn.cuda(GPUID[0]), z_random_2.cuda(GPUID[0]))

        # Inverse
        c_ht_inv_rand_1 = G(h_img.cuda(GPUID[0]), bw_ht_edge.cuda(GPUID[0]), z_random_1.cuda(GPUID[0]))
        c_ht_inv_rand_2 = G(h_img.cuda(GPUID[0]), bw_ht_edge.cuda(GPUID[0]), z_random_2.cuda(GPUID[0]))
        for channel in range(3):
            if channel == 0:
                c_img_inv_rand_1 = IHN(c_ht_inv_rand_1[:, channel, :, :].cuda(GPUID[0]).unsqueeze(1), z=noise.cuda(GPUID[0]))[-1]
                c_img_inv_rand_2 = IHN(c_ht_inv_rand_2[:, channel, :, :].cuda(GPUID[0]).unsqueeze(1), z=noise.cuda(GPUID[0]))[-1]
            else:
                c_img_inv_rand_1 = torch.cat([c_img_inv_rand_1, IHN(c_ht_inv_rand_1[:, channel, :, :].cuda(GPUID[0]).unsqueeze(1), z=noise.cuda(GPUID[0]))[-1] ], 1)
                c_img_inv_rand_2 = torch.cat([c_img_inv_rand_2, IHN(c_ht_inv_rand_2[:, channel, :, :].cuda(GPUID[0]).unsqueeze(1), z=noise.cuda(GPUID[0]))[-1] ], 1)

        # Fuse
        c_img_rand_1 = F(c_img_forward_rand_1.cuda(GPUID[0]), c_img_inv_rand_1.cuda(GPUID[0]))
        c_img_rand_2 = F(c_img_forward_rand_2.cuda(GPUID[0]), c_img_inv_rand_2.cuda(GPUID[0]))
        
        # Save Result
        save_image(h_img, '%s%d_real_h.jpg' % (args.result_save_dir, step))
        save_image(ref_img, '%s%d_real_ref.jpg' % (args.result_save_dir, step))
        save_image(c_img_ref, '%s%d_ref_c.jpg' % (args.result_save_dir, step))
        save_image(c_img_rand_1, '%s%d_rand_c_1.jpg' % (args.result_save_dir, step))
        save_image(c_img_rand_2, '%s%d_rand_c_2.jpg' % (args.result_save_dir, step))
