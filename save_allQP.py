import torch
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from skimage.transform import resize
from collections import OrderedDict
from Network import Net
import utils
from tqdm import tqdm

# 模型路径
ckp_path = 'exp/STDR/ckp_300000.pt'


def load_model():
    """加载预训练模型"""
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
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # 去除 module 前缀
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()
    print(f'> 模型 {ckp_path} 已加载.')
    return model


def yuv420_to_rgb(y, u, v):
    """将YUV420转换为RGB"""
    u_up = resize(u, y.shape, mode='reflect', anti_aliasing=True)
    v_up = resize(v, y.shape, mode='reflect', anti_aliasing=True)
    y = y * 255.0
    u_up = u_up * 255.0 - 128.0
    v_up = v_up * 255.0 - 128.0
    r = y + 1.402 * v_up
    g = y - 0.344136 * u_up - 0.714136 * v_up
    b = y + 1.772 * u_up
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


def save_frame(y, u, v, path):
    """保存帧为PNG"""
    from PIL import Image
    Image.fromarray(yuv420_to_rgb(y, u, v)).save(path)


def process_video(model, raw_path, lq_path, h, w, nfs, output_dir):
    """处理单个视频"""
    # 创建输出目录
    enh_dir = os.path.join(output_dir, "enh_frames")
    os.makedirs(enh_dir, exist_ok=True)

    # 加载YUV数据
    try:
        raw_y, raw_u, raw_v = utils.import_yuv(raw_path, h, w, nfs, 0, False)
        lq_y, lq_u, lq_v = utils.import_yuv(lq_path, h, w, nfs, 0, False)
    except Exception as e:
        print(f"加载YUV失败: {e}")
        return

    raw_y = raw_y.astype(np.float32) / 255.
    lq_y = lq_y.astype(np.float32) / 255.

    # 处理帧
    pbar = tqdm(total=min(nfs, 50), desc=os.path.basename(lq_path), ncols=100)
    for idx in range(min(nfs, 50)):
        idx_list = np.clip(range(idx - 3, idx + 4), 0, nfs - 1)
        input_data = torch.stack([torch.from_numpy(lq_y[i]).cuda() for i in idx_list]).unsqueeze(0)

        with torch.no_grad():
            enhanced = model(input_data)

        save_frame(enhanced[0, 0].cpu().numpy(), lq_u[idx], lq_v[idx],
                   os.path.join(enh_dir, f"frame_{idx:04d}.png"))

        del input_data, enhanced
        torch.cuda.empty_cache()
        pbar.update()
    pbar.close()


def main():
    model = load_model()

    # 配置路径
    raw_base = '/data/lk/datasets/MFQEv2_dataset/test_18/raw/'
    lq_base = '/data/lk/datasets/MFQEv2_dataset/test_18/HM16.5_LDP/'
    output_base = 'output_frames/'

    # 遍历所有QP目录
    for qp_folder in os.listdir(lq_base):
        qp_path = os.path.join(lq_base, qp_folder)
        if not os.path.isdir(qp_path):
            continue

        # 处理每个YUV文件
        for yuv_file in os.listdir(qp_path):
            if not yuv_file.endswith('.yuv'):
                continue

            # 解析视频参数
            try:
                name_part, res, frames = yuv_file.split('_')[:3]
                w, h = map(int, res.split('x'))
                nfs = int(frames.split('.')[0])
            except:
                print(f"参数解析失败: {yuv_file}")
                continue

            # 构建路径
            raw_yuv = os.path.join(raw_base, yuv_file)
            lq_yuv = os.path.join(qp_path, yuv_file)
            if not os.path.exists(raw_yuv):
                print(f"缺失原始文件: {raw_yuv}")
                continue

            # 创建输出目录
            output_dir = os.path.join(output_base, qp_folder, yuv_file[:-4])

            # 处理视频
            print(f"\n处理视频: {yuv_file} (分辨率 {w}x{h}, 帧数 {nfs})")
            process_video(model, raw_yuv, lq_yuv, h, w, nfs, output_dir)

    print("\n所有视频处理完成!")


if __name__ == '__main__':
    main()