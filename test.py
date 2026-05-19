import argparse
import cv2
import torch
import scipy.io as scio
import os

from net import Net
from utils.utils import seed_pytorch, get_optimizer
from utils.datasets import NUDTSIRSTSetLoader
from utils.datasets import IRSTD1KSetLoader
from utils.datasets import SIRSTAugSetLoader
# from utils.datasets import SIRSTSetLoader
from torch.autograd import Variable
from torch.utils.data import DataLoader
from parser_test import parse_args as parser
import numpy as np
# parser = argparse.ArgumentParser(description="PyTorch L2SKNet test")
#
# parser.add_argument("--model_names", default='L2SKNet_UNet', type=str,  # nargs='+',
#                     help="model_name: 'L2SKNet_UNet', 'L2SKNet_FPN', "
#                          "'L2SKNet_1D_UNet', 'L2SKNet_1D_FPN'")
# parser.add_argument("--dataset_names", default='NUDT-SIRST', type=str,  # nargs='+',
#                     help="dataset_name: 'NUDT-SIRST', 'IRSTD-1K', 'SIRST-aug','SIRST','NUAA-SIRST'")
# parser.add_argument("--dataset_dir", default='./data', type=str, help="train_dataset_dir")
# parser.add_argument("--save", default='./log', type=str, help="Save path of checkpoints")
# parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
# parser.add_argument("--test_epo", type=str, default='200', help="Number of epoch for test")

# global opt
# opt = parser.parse_args()
# seed_pytorch(opt.seed)

class Trainer(object):
    def __init__(self, opt):
        self.opt = opt

    def test(self):
        if (opt.dataset_name[:10] == "NUDT-SIRST"):
            dataset_dir = r'./data/NUDT-SIRST/'
            test_set = NUDTSIRSTSetLoader(base_dir=dataset_dir, mode='test')
        elif (opt.dataset_name[:8] == "IRSTD-1K"):
            dataset_dir = r'./data/IRSTD-1K/'
            test_set = IRSTD1KSetLoader(base_dir=dataset_dir, mode='test')
        elif (opt.dataset_name == "SIRST-aug"):
            dataset_dir = r'./data/sirst_aug/'
            test_set = SIRSTAugSetLoader(base_dir=dataset_dir, mode='test')
        else:
            raise NotImplementedError

        param_path = "log/" + opt.dataset_name + "/" + opt.model_name + '/' + opt.test_epo + '.pth.tar'

        test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

        net = Net(model_name=opt.model_name).cuda(device=0)

        net.load_state_dict(torch.load(param_path, map_location='cuda:0')['state_dict'], False)
        net.eval()

        print('testing data=' + opt.dataset_name + ', model=' + opt.model_name + ', epoch=' + opt.test_epo)

        imgDir = "./result/" + opt.dataset_name + "/img/" + opt.model_name + "/"
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
        matDir = "./result/" + opt.dataset_name + "/mat/" + opt.model_name + "/"
        if not os.path.exists(matDir):
            os.makedirs(matDir)

        for idx_iter, (img, gt_mask, size, iname) in enumerate(test_loader):
            name = iname[0]
            pngname = name + ".png"
            matname = name + '.mat'
            with torch.no_grad():
                # img = img.to('cpu')
                # np.save('img.npy', img)
                img = Variable(img).cuda(device=0)
                pred = net.forward(img)
                pred = pred[:, :, :size[0], :size[1]]
                pred_out = pred.data.cpu().detach().numpy().squeeze()
                # np.save("test_final.npy", pred_out)
                pred_out_png = pred_out * 255

            # cv2.imwrite(imgDir + pngname, pred_out_png)
            # scio.savemat(matDir + matname, {'T': pred_out})

            # if idx_iter == 0:
            #     break


def main(opt):
    seed_pytorch(opt.seed)
    trainer = Trainer(opt)
    trainer.test()


if __name__ == '__main__':
    # for dataset_name in opt.dataset_names:
    #     opt.dataset_name = dataset_name
    #     for model_name in opt.model_names:
    #         opt.model_name = model_name
    #         if not os.path.exists(opt.save):
    #             os.makedirs(opt.save)
    #         test()
    #         print('\n')
    opt = parser()
    main(opt)


# import argparse
# import cv2
# import torch
# import scipy.io as scio
# import os
#
# from net import Net
# from utils.utils import seed_pytorch, get_optimizer
# from utils.datasets import NUDTSIRSTSetLoader
# from utils.datasets import IRSTD1KSetLoader
# from utils.datasets import SIRSTAugSetLoader
# # from utils.datasets import SIRSTSetLoader
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
#
# from evaluation.mIoU import mIoU
# from evaluation.pd_fa import PD_FA
# from evaluation.TPFNFP import SegmentationMetricTPFNFP
#
# parser = argparse.ArgumentParser(description="PyTorch L2SKNet test")
#
# parser.add_argument("--model_names", default='L2SKNet_UNet', type=str, nargs='+',
#                     help="model_name: 'L2SKNet_UNet', 'L2SKNet_FPN', "
#                          "'L2SKNet_1D_UNet', 'L2SKNet_1D_FPN'")
# parser.add_argument("--dataset_names", default='NUDT-SIRST', type=str, nargs='+',
#                     help="dataset_name: 'NUDT-SIRST', 'IRSTD-1K', 'SIRST-aug','SIRST','NUAA-SIRST'")
# parser.add_argument("--dataset_dir", default='./data', type=str, help="train_dataset_dir")
# parser.add_argument("--save", default='./log', type=str, help="Save path of checkpoints")
# parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
# parser.add_argument("--test_epo", type=str, default='200', help="Number of epoch for test")
# parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
#
# global opt
# opt = parser.parse_args()
# seed_pytorch(opt.seed)
#
#
# def test():
#     if (opt.dataset_name[:10] == "NUDT-SIRST"):
#         dataset_dir = r'./data/NUDT-SIRST/'
#         test_set = NUDTSIRSTSetLoader(base_dir=dataset_dir, mode='test')
#     elif (opt.dataset_name[:8] == "IRSTD-1K"):
#         dataset_dir = r'./data/IRSTD-1K/'
#         test_set = IRSTD1KSetLoader(base_dir=dataset_dir, mode='test')
#     elif (opt.dataset_name == "SIRST-aug"):
#         dataset_dir = r'./data/sirst_aug/'
#         test_set = SIRSTAugSetLoader(base_dir=dataset_dir, mode='test')
#     else:
#         raise NotImplementedError
#
#     param_path = "log/" + opt.dataset_name + "/" + opt.model_name + '/' + opt.test_epo + '.pth.tar'
#
#     test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
#
#     net = Net(model_name=opt.model_name).cuda(device=0)
#
#     net.load_state_dict(torch.load(param_path, map_location='cuda:0')['state_dict'], False)
#     net.eval()
#
#     print('testing data=' + opt.dataset_name + ', model=' + opt.model_name + ', epoch=' + opt.test_epo)
#
#     # imgDir = "./result/" + opt.dataset_name + "/img/" + opt.model_name + "/"
#     # if not os.path.exists(imgDir):
#     #     os.makedirs(imgDir)
#     # matDir = "./result/" + opt.dataset_name + "/mat/" + opt.model_name + "/"
#     # if not os.path.exists(matDir):
#     #     os.makedirs(matDir)
#
#     # for idx_iter, (img, gt_mask, size, iname) in enumerate(test_loader):
#     #     name = iname[0]
#     #     pngname = name + ".png"
#     #     matname = name + '.mat'
#     #     with torch.no_grad():
#     #         img = Variable(img).cuda(device=0)
#     #         pred = net.forward(img)
#     #         pred = pred[:, :, :size[0], :size[1]]
#     #         pred_out = pred.data.cpu().detach().numpy().squeeze()
#     #         pred_out_png = pred_out * 255
#
#     #     cv2.imwrite(imgDir + pngname, pred_out_png)
#     #     scio.savemat(matDir + matname, {'T': pred_out})
#
#     eval_mIoU = mIoU()
#     eval_PD_FA = PD_FA()
#     eval_mIoU_P_R_F = SegmentationMetricTPFNFP(nclass=1)
#
#     for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
#         with torch.no_grad():
#             img = Variable(img).cuda(device=0)
#             pred = net.forward(img)
#             pred = pred[:, :, :size[0], :size[1]]
#
#         gt_mask = gt_mask[:, :, :size[0], :size[1]]
#
#         eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask)
#         eval_PD_FA.update(pred[0, 0, :, :].cpu().detach().numpy(), gt_mask[0, 0, :, :].detach().numpy(), size)
#         eval_mIoU_P_R_F.update(labels=gt_mask[0, 0, :, :].detach().numpy(),
#                                preds=pred[0, 0, :, :].cpu().detach().numpy())
#
#     Ying_pixAcc, Ying_mIoU = eval_mIoU.get()
#     pd, fa = eval_PD_FA.get()
#     _, _, _, fscore = eval_mIoU_P_R_F.get()
#     print('pixAcc %.6f, mIoU: %.6f' % (Ying_pixAcc, Ying_mIoU))
#     print('Pd: %.6f, Fa: %.8f, fscore: %.6f' % (pd, fa, fscore))
#
#
# if __name__ == '__main__':
#     for dataset_name in opt.dataset_names:
#         opt.dataset_name = dataset_name
#         for model_name in opt.model_names:
#             opt.model_name = model_name
#             if not os.path.exists(opt.save):
#                 os.makedirs(opt.save)
#             test()
#             print('\n')
