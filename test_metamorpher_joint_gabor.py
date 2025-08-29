
import os
from torch.autograd import Variable

from scipy.misc import imread, imsave

from gaborconv_opti_test import gabor_net
import numpy as np
from img_read_save import image_read_cv2
from Evaluator import Evaluator
import torch
import  math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
total_time = 0

import argparse
import os

def _check_dir(path: str):
    """确保目录存在，否则自动创建"""
    os.makedirs(path, exist_ok=True)
    return path

def get_args():
    parser = argparse.ArgumentParser(
        description="Testing for MetaMorph_MuF_joint_gabor"
    )

    # ========= 路径相关 =========
    parser.add_argument("--test_root", type=str,
                        default="test_imgs/MSRS",
                        help="directory for testing set")
    parser.add_argument("--model_dir", type=str,
                        default="weights/MetaMorph_MuF_joint_gabor/MetaMorph_MuF_joint_gabor.model",
                        help="directory for the restored model")
    parser.add_argument("--out_root", type=str,
                        default="outputs/MetaMorph_MuF_joint_gabor/fused_imgs",
                        help="directory for restoring the fused images")
    parser.add_argument("--result_file", type=str,
                        default="outputs/MetaMorph_MuF_joint_gabor/result.txt",
                        help="")

    # ========= 运行参数 =========
    parser.add_argument("--cuda", type=int, default=0,
                        help="0 表示CPU，1 表示 GPU")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU id")
    parser.add_argument("--num_imgs", type=int, default=361,
                        help="the number of testing images")

    # ========= 网络结构 =========
    parser.add_argument("--in_channels", type=int, default=2)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--middle_channels", type=int, default=4)



    args = parser.parse_args()

    # 后处理：自动创建目录
    _check_dir(args.out_root)

    return args




def gabor_real(sigma, theta, lamda, gamma, psi,args):
    sigma_x = sigma
    sigma_y = sigma / gamma
    xmax = ymax = 3 // 2
    xmin, ymin = -xmax, -ymax
    x = torch.arange(xmin, xmax + 1, device=args.device)
    y = torch.arange(ymin, ymax + 1, device=args.device)
    x, y = torch.meshgrid(x, y)
    c = torch.cos(theta)
    s = torch.sin(theta)
    x_alpha = x * c + y * s
    y_alpha = -x * s + y * c
    exponent = torch.exp(-.5 * (x_alpha ** 2 / sigma_x ** 2 + y_alpha ** 2 / sigma_y ** 2))
    kernel = exponent * torch.cos(2 * torch.tensor(math.pi, device=args.device) / lamda * x_alpha + psi)
    norm = torch.norm(kernel, p=2)  # 计算L2范数
    kernel = kernel / norm  # 归一化核
    return kernel

def G(sigma, theta, lamda, gamma, psi, shape,args):
    gabor_filter = torch.empty(shape, device=args.device)
    for i in range(shape[0]):
        for j in range(shape[1]):
            param_idx = i * shape[1] + j
            if param_idx >= len(sigma):
                raise IndexError("Parameter index out of bounds")
            gabor_filter[i, j] = gabor_real(sigma[param_idx], theta[param_idx], lamda[param_idx], gamma[param_idx],
                                                 psi[param_idx],args)
    return gabor_filter

def changetoconv(sigma, theta, lamda, gamma, psi,args):
    change_lambda = torch.exp(lamda) + 2
    change_gamma = torch.exp(gamma) + 1e-6
    change_sigmma = torch.exp(sigma) + 1e-6
    conv1_weight = G(change_sigmma[:8], theta[:8], change_lambda[:8], change_gamma[:8], psi[:8],
                          shape=(4, 2, 3, 3),args=args)
    conv2_weight = G(change_sigmma[8:], theta[8:], change_lambda[8:], change_gamma[8:], psi[8:],
                          shape=(1, 4, 3, 3),args=args)
    return conv1_weight, conv2_weight



def load_model(path, input_nc, output_nc,middle_nc,args):
    nest_model = gabor_net(input_nc, output_nc,middle_nc)
    print(gabor_net)
    nest_model.load_state_dict(torch.load(path))
    sig = nest_model.sigma
    the = nest_model.theta
    gam = nest_model.gamma
    lam = nest_model.lamda
    psi = nest_model.psi
    conv1, conv2 = changetoconv(sig, the, lam, gam, psi,args)
    for i in range(12):
        print("第{}个gabor滤波器的参数为：".format(i + 1))
        print("sigma:{},theta:{},lambda:{},gamma:{},psi:{}".format(sig[i], the[i], lam[i], gam[i], psi[i]))

    total_param = 0
    print("MODEL DETAILS:\n")
    for param in nest_model.parameters():
        print(param.dtype)  # 查看每个参数大小
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', nest_model._get_name(), total_param)
    #
    # 每个参数是一个 64 位浮点数（8 字节）
    bytes_per_param = 4

    # 计算总字节数
    total_bytes = total_param * bytes_per_param

    # 转换为兆字节（MB）和千字节（KB）
    total_megabytes = total_bytes / (1024 * 1024)
    total_kilobytes = total_bytes / 1024

    print("Total parameters in MB:", total_megabytes)
    print("Total parameters in KB:", total_kilobytes)
    nest_model.eval()
    return nest_model,conv1,conv2


def run_demo(model, infrared_path, visible_path, output_path_root, index, metrics, args,conv1,conv2):
    ir_img = imread(infrared_path, mode='L');  # mode='L'代表8位像素，黑白
    vi_img = imread(visible_path, mode='L');
    ir_img = ir_img / 255.0;
    vi_img = vi_img / 255.0;
    ir_img_patches = [[ir_img]]
    vi_img_patches = [[vi_img]]

    ir_img_patches = np.stack(ir_img_patches, axis=0);
    vi_img_patches = np.stack(vi_img_patches, axis=0);
    ir_img_patches = torch.from_numpy(ir_img_patches);
    vi_img_patches = torch.from_numpy(vi_img_patches);

    # dim = img_ir.shape
    if args.cuda:
        ir_img_patches = ir_img_patches.cuda(args.device)
        vi_img_patches = vi_img_patches.cuda(args.device)
        model = model.cuda(args.device);
    ir_img_patches = Variable(ir_img_patches, requires_grad=False)
    vi_img_patches = Variable(vi_img_patches, requires_grad=False)

    img = torch.cat([ir_img_patches, vi_img_patches], 1);
    img = img.float()


    en = model.forward(img,conv1,conv2)

    out = (en[0][0][0]).detach().cpu().numpy();

    #
    # ########################### multi outputs_LLVIP ##############################################
    #
    # print(out.shape)  # (99,2,128,128)
    #
    file_name = str(index) + '.png'
    output_path = os.path.join(output_path_root, file_name)
    print(output_path)
    imsave(output_path, out)


    ir = image_read_cv2(infrared_path, 'GRAY')
    print("红外形状：{}".format(ir.shape))
    vi = image_read_cv2(visible_path, 'GRAY')
    print("可见光形状：{}".format(ir.shape))
    fi = image_read_cv2(output_path, 'GRAY')
    print("融合图像形状：{}".format(ir.shape))
    metrics += np.array([Evaluator.EN(fi), Evaluator.SD(fi), Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                            , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi), Evaluator.Qabf(fi, ir, vi)
                            , Evaluator.SSIM(fi, ir, vi), Evaluator.CC(fi, ir, vi)])
    return metrics


def main():
    args = get_args()
    # 选择设备
    if args.device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    metric_result = np.zeros(9)

    model_path = args.model_dir
    out_dir = args.out_root
    os.makedirs(out_dir, exist_ok=True)


    model, conv1, conv2 = load_model(model_path, args.in_channels,args.out_channels, middle_nc=4,args=args)
    model.to(device)
    for idx in range(1, args.num_imgs + 1):
        metrics = np.zeros((9))
        ir_path = os.path.join(args.test_root, "IR", f"{idx}.png")
        vi_path = os.path.join(args.test_root, "VIS", f"{idx}.png")
        metric_result += run_demo(model,
                            ir_path,
                            vi_path,
                            out_dir,
                            idx,
                            metrics,
                            args,
                            conv1,conv2)
    metric_result /= args.num_imgs
    with open(args.result_file, "a", encoding="utf-8") as f:
        res = ('\t'.join(f"{np.round(metric_result[i], 2)}" for i in range(9)))
        f.write("\n")
        f.write(res)
if __name__ == '__main__':
    main()
