# -*- coding: utf-8 -*-
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from config.setting import configs
from dataset.dataset import MyDataset, SegmentationPresetTest
from utils import set_seed, test_with_per_image, compute_gray

from models import (
    SwinUnet,
    Unet,
    PSPNet,
    unet3up,
    UNet5,
    VMUnet3p,
    GCNUnet,
    gbcu,
    vmuccm,
    VMUNet,
    VMUNetV2,
    gbcuold,
    gbcu1,
    gbcusk,
    gbcmu,
    unetplse,
    mgbcu,
    Unet5gbc,
    Unet5sk,
    Unet5gbcsk,
    DenseUNet,
    H_vmunet,
    mbgcu
)

def build_model(name: str, num_classes: int, default_model_config: dict):
    """
    根据名称构造模型，保持与你原来 if-elif 等价。
    """
    if name == 'SwinUnet':
        return SwinUnet(img_size=224, num_classes=num_classes, drop=0.1, attn_drop=0.1)
    elif name == 'Unet':
        return Unet(input_channels=3, num_classes=num_classes)
    elif name == 'PSPNet':
        return PSPNet(num_classes=num_classes)
    elif name == 'UNet5':
        return UNet5(input_channels=3, out_channels=num_classes)
    elif name == 'unet3up':
        return unet3up(input_channels=3, n_classes=num_classes)
    elif name == 'GCNUnet':
        return GCNUnet(**default_model_config)
    elif name == 'VMUnet3p':
        return VMUnet3p(**default_model_config)
    elif name == 'gbcu':
        return gbcu(**default_model_config)
    elif name == 'vmuccm':
        return vmuccm(**default_model_config)
    elif name == 'VMUNetV2':
        return VMUNetV2(**default_model_config)
    elif name == 'VMUNet':
        return VMUNet(**default_model_config)
    elif name == 'gbcuold':
        return gbcuold(**default_model_config)
    elif name == 'gbcu1':
        return gbcu1(**default_model_config)
    elif name == 'gbcusk':
        return gbcusk(**default_model_config)
    elif name == 'gbcmu':
        return gbcmu(**default_model_config)
    elif name == 'unetplse':
        return unetplse(**default_model_config)
    elif name == 'mgbcu':
        return mgbcu(**default_model_config)
    elif name == 'Unet5gbc':
        return Unet5gbc(input_channels=3, out=num_classes)
    elif name == 'Unet5sk':
        return Unet5sk(input_channels=3, out=num_classes)
    elif name == 'Unet5gbcsk':
        return Unet5gbcsk(input_channels=3, out=num_classes)
    elif name == 'DenseUNet':
        return DenseUNet(n_classes=num_classes)
    elif name == 'H_vmunet':
        return H_vmunet(num_classes=num_classes, input_channels=3)
    elif name == 'mbgcu':
        return mbgcu(**default_model_config)
    else:
        raise ValueError(f"未知模型：{name}")
# =============== 模型列表 ===============
ALL_MODEL_NAMES = [
    'SwinUnet', 'Unet', 'PSPNet', 'unet3up', 'UNet5', 'VMUnet3p', 'GCNUnet', 'gbcu',
    'vmuccm', 'VMUNet', 'VMUNetV2', 'gbcuold', 'gbcu1', 'gbcusk', 'gbcmu', 'unetplse',
    'mgbcu', 'Unet5gbc', 'Unet5sk', 'Unet5gbcsk', 'DenseUNet', 'H_vmunet', 'mbgcu'
]
default_model_config = {
        'num_classes': 2,
        'input_channels': 3,
        'depths': [2,2,9,2],
        'depths_decoder': [2,2,2,1],
        'drop_path_rate': 0.2,
        'load_ckpt_path': './pre_trained_weights/vmamba_small_e238_ema.pth',
        'deep_supervision': False,
    }

def test_single_model(name, args, result_root_dir):
    """
    测试单个模型并保存结果。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Testing model: {name} on {device} ===")
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dataname = args.data
    print(dataname)
    # 权重路径（固定逻辑）
    weights_path = f'./new_run_results/{name}_{dataname}/best_model.pth'
    if not os.path.isfile(weights_path):
        print(f"权重未找到，跳过: {weights_path}")
        return None

    # 构造保存路径（每个模型单独文件夹）
    model_result_dir = os.path.join(result_root_dir, name)
    per_image_dir = os.path.join(model_result_dir, 'per_image_metrics')
    vis_image_dir = os.path.join(model_result_dir, 'vis_images')  # ✅ 新增
    os.makedirs(per_image_dir, exist_ok=True)
    os.makedirs(vis_image_dir, exist_ok=True)  # ✅ 新增

    # 测试集准备
    test_img_dir = f'./data/{args.data}/test/images'
    test_transform = SegmentationPresetTest()
    test_dataset = MyDataset(imgs_path=test_img_dir, txt_path=args.txt_path, transform=test_transform)
    num_workers = min(os.cpu_count(), args.batch_size, 8)
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=num_workers, shuffle=False)

    # 类别数计算
    num_classes = compute_gray(test_img_dir.replace('images', 'masks'))
    print(f"[num_classes] {num_classes}")

    # 模型构建与加载
    model = build_model(name, num_classes, default_model_config)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.to(device)
    model.eval()

    # 测试并保存逐图像 CSV
    per_image_csv = os.path.join(per_image_dir, f"{name}_per_image.csv")
    try:
        test_miou, _, _ = test_with_per_image(
            model=model,
            test_loader=test_loader,
            device=device,
            num_classes=num_classes,
            save_csv_path=per_image_csv,
            ignore_index=255,
            save_vis_dir=vis_image_dir  # ✅ 传入可视化图像保存路径
        )
        print(f"{name} mean IoU: {test_miou:.4f}")
        return test_miou
    except Exception as e:
        print(f"测试失败 {name}: {e}")
        return None


def main(args):
    results = []
    result_root_dir = args.save_dir or f'./test_all_results/{args.data}'  # 保存结果路径
    os.makedirs(result_root_dir, exist_ok=True)

    if args.test_all:
        for name in ALL_MODEL_NAMES:
            miou = test_single_model(name, args, result_root_dir)
            if miou is not None:
                results.append({'model': name, 'mIoU': miou})

        # 保存汇总
        if results:
            import pandas as pd
            summary_csv = os.path.join(result_root_dir, 'summary.csv')
            pd.DataFrame(results).to_csv(summary_csv, index=False)
            print(f"\n所有模型测试完成，汇总保存在 {summary_csv}")
    else:
        miou = test_single_model(args.network, args, result_root_dir)
        if miou is None:
            print(f"模型 {args.network} 测试未成功。请检查权重路径或模型名称。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="仅测试脚本，支持单模型或所有模型测试")
    parser.add_argument('--test-all', action='store_true', help="测试 ALL_MODEL_NAMES 中的所有模型，忽略 --network")
    parser.add_argument('--network', default='VMUNetV2', type=str, help="单模型测试时的模型名称")
    parser.add_argument('--data', default='new622', type=str, help="数据集名称，对应 ./data/<data>")
    parser.add_argument('--txt-path', default='./data/grayList.txt', type=str, help="灰度列表路径，用于数据预处理")
    parser.add_argument('--save-dir', default=None, type=str, help="测试结果保存根目录，如 ./test_all_results")
    parser.add_argument('--batch-size', default=8, type=int, help="测试时 DataLoader 的 batch_size 限制 num_workers")
    parser.add_argument('--seed', default=42, type=int, help="随机种子")
    args = parser.parse_args()
    main(args)
