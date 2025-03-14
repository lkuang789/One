import torch
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from skimage.transform import resize
from collections import OrderedDict
from Network import Net
import utils
from tqdm import tqdm

# 模型路径和视频路径
ckp_path = 'exp/STDR/QP37/ckp_300000.pt'  # 预训练模型路径
raw_yuv_path = '/data/lk/datasets/MFQEv2_dataset/test_18/raw/BasketballDrive_1920x1080_500.yuv'
lq_yuv_path = '/data/lk/datasets/MFQEv2_dataset/test_18/HM16.5_LDP/QP37/BasketballDrive_1920x1080_500.yuv'
h, w, nfs = 1080, 1920, 500  # 视频分辨率和帧数

# 输出目录
output_dir = "output_frames/QP37/BasketballDrive_1920x1080_500"
lq_dir = os.path.join(output_dir, "lq_frames")
enh_dir = os.path.join(output_dir, "enh_frames")
raw_dir = os.path.join(output_dir, "raw_frames")
# os.makedirs(lq_dir, exist_ok=True)
os.makedirs(enh_dir, exist_ok=True)
# os.makedirs(raw_dir, exist_ok=True)


def yuv420_to_rgb(y, u, v):
    """
    将 YUV420 格式转换为 RGB 图像。
    Args:
        y (ndarray): Y 通道 (h, w)
        u (ndarray): U 通道 (h/2, w/2)
        v (ndarray): V 通道 (h/2, w/2)
    Returns:
        rgb (ndarray): RGB 图像 (h, w, 3)
    """
    u_up = resize(u, y.shape, mode='reflect', anti_aliasing=True)
    v_up = resize(v, y.shape, mode='reflect', anti_aliasing=True)

    y = y * 255.0
    u_up = u_up * 255.0 - 128.0
    v_up = v_up * 255.0 - 128.0

    r = y + 1.402 * v_up
    g = y - 0.344136 * u_up - 0.714136 * v_up
    b = y + 1.772 * u_up

    rgb = np.stack((r, g, b), axis=-1).clip(0, 255).astype(np.uint8)
    return rgb

def save_frame(y, u, v, path):
    """保存帧为 RGB 图像"""
    rgb_frame = yuv420_to_rgb(y, u, v)
    from PIL import Image
    Image.fromarray(rgb_frame).save(path)

def main():
    # ========== 加载预训练模型 ==========
    opts_dict = {
        'radius': 3,
        'stdf': {
            'in_nc': 1,
            'out_nc': 64,
            'nf': 64,
            'nb': 3,
            'base_ks': 3,
            'deform_ks': 3,
        },
        'Ada_RDBlock_num': 4,
        'Ada_RDBlock': {
            'in_nc': 32,
            'growthRate': 32,
            'num_layer': 4,
            'reduce_ratio': 4,
            'a': 1,
            'b': 0.2
        },
    }
    model = Net(opts_dict=opts_dict)
    print(f'加载模型 {ckp_path}...')
    checkpoint = torch.load(ckp_path)
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # 多 GPU 训练模型
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # 去除 module 前缀
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # 单 GPU 模型
        model.load_state_dict(checkpoint['state_dict'])

    print(f'> 模型 {ckp_path} 已加载.')
    model = model.cuda()
    model.eval()

    # ========== 加载 YUV 视频 ==========
    print(f'加载 YUV 视频...')
    raw_y, raw_u, raw_v = utils.import_yuv(
        seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False
    )
    lq_y, lq_u, lq_v = utils.import_yuv(
        seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False
    )
    raw_y = raw_y.astype(np.float32) / 255.
    lq_y = lq_y.astype(np.float32) / 255.
    print('> 视频加载完成.')

    # ========== 定义 PSNR ==========
    criterion = utils.PSNR()
    unit = 'dB'

    # 预先将所有数据加载到 GPU 中
    raw_y_tensor = torch.from_numpy(raw_y).cuda()
    lq_y_tensor = torch.from_numpy(lq_y).cuda()

    pbar = tqdm(total=min(nfs, 200), ncols=80)
    # ori_psnr_counter = utils.Counter()
    # enh_psnr_counter = utils.Counter()

    for idx in range(min(nfs, 200)):
        # 提取输入帧（前后帧）
        idx_list = list(range(idx - 3, idx + 4))
        idx_list = np.clip(idx_list, 0, nfs - 1)
        input_data = []
        for idx_ in idx_list:
            input_data.append(lq_y_tensor[idx_])
        input_data = torch.stack(input_data)  # 批量堆叠为一个 Tensor
        input_data = input_data.unsqueeze(0).cuda()  # 扩展维度并加载到 GPU

        # 增强帧
        enhanced_frm = model(input_data)

        # 获取对应的 ground truth
        gt_frm = raw_y_tensor[idx]

        # 计算 PSNR
        # batch_ori = criterion(input_data[0, 3, ...], gt_frm)
        # batch_perf = criterion(enhanced_frm[0, 0, ...], gt_frm)
        # ori_psnr_counter.accum(volume=batch_ori)
        # enh_psnr_counter.accum(volume=batch_perf)

        # 保存帧
        # save_frame(lq_y[idx], lq_u[idx], lq_v[idx], os.path.join(lq_dir, f"frame_{idx:04d}.png"))
        save_frame(enhanced_frm[0, 0, ...].detach().cpu().numpy(), lq_u[idx], lq_v[idx], os.path.join(enh_dir, f"frame_{idx:04d}.png"))
        # save_frame(raw_y[idx], raw_u[idx], raw_v[idx], os.path.join(raw_dir, f"frame_{idx:04d}.png"))

        # 显示进度
        # pbar.set_description(
        #     "[{:.3f}] {:s} -> [{:.3f}] {:s}"
        #     .format(batch_ori, unit, batch_perf, unit)
        # )
        pbar.update()

    pbar.close()

    # ori_ = ori_psnr_counter.get_ave()
    # enh_ = enh_psnr_counter.get_ave()
    # print('平均 PSNR 原始 [{:.3f}] {:s}, 增强 [{:.3f}] {:s}, 增量 [{:.3f}] {:s}'.format(
    #     ori_, unit, enh_, unit, (enh_ - ori_), unit
    # ))
    print('> 完成.')


if __name__ == '__main__':
    main()
