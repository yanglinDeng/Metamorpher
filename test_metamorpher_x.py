
import os
from torch.autograd import Variable

from scipy.misc import imread, imsave
from stunet3 import stu_net3
import numpy as np
from img_read_save import image_read_cv2
from Evaluator import Evaluator
import torch
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
                        default="est_imgs/MSRS",
                        help="directory for testing set")
    parser.add_argument("--model_dir", type=str,
                        default="weights/MetaMorph_MuF_joint/MetaMorph_MuF_joint.model",
                        help="directory for the restored model")
    parser.add_argument("--out_root", type=str,
                        default="outputs/MetaMorph_MuF_joint/fused_imgs",
                        help="directory for restoring the fused images")
    parser.add_argument("--result_file", type=str,
                        default="outputs/MetaMorph_MuF_joint/result.txt",
                        help="")
    parser.add_argument("--show_middle", type=str,
                        default="true",
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


def load_model(path, input_nc, output_nc):
    nest_model = stu_net3(input_nc, output_nc)
    nest_model.load_state_dict(torch.load(path))
    nest_model.float()
    f1 = nest_model.conv1[1].weight.data
    f2 = nest_model.conv2[1].weight.data
    b1 = nest_model.conv1[1].bias.data
    b2 = nest_model.conv2[1].bias.data
    print(f1)
    print(b1)
    print(f2)
    print(b2)
    total_param = 0
    # print("MODEL DETAILS:\n")
    for param in nest_model.parameters():
        print(param.dtype)  # 查看每个参数大小
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', nest_model._get_name(), total_param)
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

    return nest_model


def run_demo(model, infrared_path, visible_path, output_path_root, index, metrics,  args,flag):
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


    en = model.encoder(img)
    # print("%%%%",en[0].shape)


    # 保留第一组特征图
    if flag=="true":
        f_en = (en[0][0]).detach().cpu().numpy()
        for j in range(1, 5):
            r_path =  output_path_root + "middle_"+str(j)
            if os.path.exists(r_path) is False:
                os.mkdir(r_path)
            spath = r_path + "/" + str(index) + '.png'
            imsave(spath, f_en[j - 1]);

    out = (model.decoder(en)[0][0][0]).detach().cpu().numpy();

    #
    # ########################### multi outputs_LLVIP ##############################################
    #
    # print(out.shape)  # (99,2,128,128)
    #
    file_name = str(index) + '.png'
    output_path = output_path_root+"fused"
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)
    output_path = os.path.join(output_path,file_name)
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

    model = load_model(model_path,args.in_channels,args.out_channels )
    model.to(device)

    for idx in range(1, args.num_imgs + 1):
        metrics = np.zeros((9))
        ir_path = os.path.join(args.test_root, "IR", f"{idx}.png")
        vi_path = os.path.join(args.test_root, "VIS", f"{idx}.png")
        flag = args.show_middle
        metric_result += run_demo(model,
                            ir_path,
                            vi_path,
                            out_dir + "/",
                            idx,
                            metrics,
                            args,
                            flag)
    metric_result /= args.num_imgs
    with open(args.result_file, "a", encoding="utf-8") as f:
        res = ('\t'.join(f"{np.round(metric_result[i], 2)}" for i in range(9)))
        f.write("\n")
        f.write(res)
if __name__ == '__main__':
    main()
