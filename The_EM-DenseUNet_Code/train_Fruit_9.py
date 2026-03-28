# -*- coding: utf-8 -*-
import os
import time
import math
import copy
import re
import argparse
import datetime
import shutil

import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler  # 学习率衰减

import pandas as pd
from torchinfo import summary

from config.setting import configs
from loss import *

# ========= models =========
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
    DenseNet,
    DenseUNet,
    DeepLab,       # DeepLabV3模型
    DenseUNet_EFA_new,
    DenseUNet_MDAF,
    DenseUNet_MDAF_EFA,
    H_vmunet,
    mbgcu
)

# ========= dataset =========
from dataset.dataset_fruit import (
    MyDataset_Fruit_9,                 # 数据预处理
    SegmentationPresetTrain,   # 训练集预处理
    SegmentationPresetVal,     # 验证集预处理
    SegmentationPresetTest     # 测试集预处理
)

# ========= utils =========
from utils import (
    compute_gray,           # mask 的灰度值
    train_one_epoch,        # 训练一个 epoch
    evaluate,               # 评价模型精度（保留）
    test,                   # 验证模型精度（整体统计）
    plot,                   # 可视化（保留）
    plot_lr_decay,          # 学习率下降曲线
    plt_loss_iou,           # 训练集 + 测试集的 loss / miou 曲线
    set_seed,               # 设置随机数种子
    get_optimizer,          # 设置优化器
    get_scheduler,          # 设置调度器
    # * 新增：逐图像/逐类别统计导出（供显著性检验）
    test_with_per_image,    # * 新增导入
    evaluate_with_hd95,
    append_log_to_excel
)

# =============== 日志辅助 ===============
import os
import pandas as pd

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# =============== 模型构造（便于维护） ===============
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
    elif name == 'DenseNet':
        return DenseNet(n_classes=num_classes)
    elif name == 'DenseUNet':
        return DenseUNet(n_classes=num_classes)
    elif name == 'DenseUNet_EFA':
        return DenseUNet_EFA(n_classes=num_classes)
    elif name == 'DenseUNet_EFA_new':
        return DenseUNet_EFA_new(n_classes=num_classes)
    elif name == 'DenseUNet_MDAF':
        return DenseUNet_MDAF(n_classes=num_classes)
    elif name == 'DenseUNet_MDAF_EFA':
        return DenseUNet_MDAF_EFA(n_classes=num_classes)
    elif name == 'H_vmunet':
        return H_vmunet(num_classes=num_classes, input_channels=3)
    elif name == 'mbgcu':
        return mbgcu(**default_model_config)
    elif name == 'DeepLab':
        return DeepLab(num_classes=num_classes)
    else:
        raise ValueError(f"未知模型：{name}")

# =============== 训练主流程 ===============
def main(args):
    if args.save_dir:
        save_path = args.save_dir
    else:
        # 添加时间戳到保存路径
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        save_path = f'./run_results/{current_time}_{args.network}_{args.data}'
    os.makedirs(save_path, exist_ok=True)

    # 其余相关路径
    fig_dir = os.path.join(save_path, 'figs')
    per_image_dir = os.path.join(save_path, "per_image_metrics")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(per_image_dir, exist_ok=True)
    log_path = os.path.join(save_path, 'train_log_results.txt')
    iou_path = os.path.join(fig_dir, 'iou.png')
    lr_path = os.path.join(fig_dir, 'lr.png')
    if not os.path.isfile(log_path):
        with open(log_path, 'w') as f:
            pass

    # ======= 2. 断点续训相关变量 =======
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device))
    set_seed(args.seed)
    torch.cuda.empty_cache()

    # 记录超参数
    with open(log_path, "a") as f:
        info = f"[train hyper-parameters: {args}]\n"
        f.write(info)

    weights_path = os.path.join(save_path, 'best_model.pth')
    full_model_file = os.path.join(save_path, "model_full.pth")
    checkpoint_path = os.path.join(save_path, 'checkpoint.pth')  # 断点续训文件


    # ========== 数据 ==========
    train_tf = SegmentationPresetTrain()
    val_tf   = SegmentationPresetVal()
    test_tf  = SegmentationPresetTest()

    dataset_name = args.data
    if dataset_name in ('622', '613', 'new622', 'n622', 'up622'):
        categories = ["背景", "水肿", "肌肉"]
    else:
        #categories = ["背景", "果", "茎"]
       categories = ["背景", "目标"]

    excel_path = "output_{}.csv".format(dataset_name)
    train_path = f'/public/home/liupengdong2024/datastes/dataset_nine/{dataset_name}/train'
    val_path = f'/public/home/liupengdong2024/datastes/dataset_nine/{dataset_name}/val'
    test_path = f'/public/home/liupengdong2024/datastes/dataset_nine/{dataset_name}/test'
    num_classes = 2
    print(f"[num_classes] {num_classes}")

    trainDataset = MyDataset_Fruit_9(imgs_path=train_path,  transform=train_tf)
    valDataset   = MyDataset_Fruit_9(imgs_path=val_path,    transform=val_tf)
    testDataset  = MyDataset_Fruit_9(imgs_path=test_path,   transform=test_tf)

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print('Using %g dataloader workers' % num_workers)

    trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, num_workers=num_workers, shuffle=True)
    valLoader   = DataLoader(valDataset,   batch_size=1,               num_workers=num_workers, shuffle=False)
    testLoader  = DataLoader(testDataset,  batch_size=1,               num_workers=num_workers, shuffle=False)

    # ========== 模型 ==========
    default_model_config = {
        'num_classes': num_classes,
        'input_channels': 3,
        'depths': [2,2,9,2],
        'depths_decoder': [2,2,2,1],
        'drop_path_rate': 0.2,
        'load_ckpt_path': './pre_trained_weights/vmamba_small_e238_ema.pth',
        'deep_supervision': False,
    }

    model = build_model(args.network, num_classes, default_model_config)
    if hasattr(model, 'load_from'):
        model.load_from()
    model.to(device)

    # ========== 优化器 / 调度器 / 损失 ==========
    optimizer = get_optimizer(configs, model)
    scheduler = get_scheduler(configs, optimizer)

    # * 注意：如果你的 plot_lr_decay 内部会对 scheduler.step()，可能影响实际训练；
    # * 如有此情况，可考虑拷贝一个 scheduler 的“演示用副本”传给绘图函数。
    plot_lr_decay(scheduler, optimizer, args.epochs, lr_path)

    if args.loss == 'CE':
        # 权重数量必须与 num_classes 一致
        class_weights = torch.tensor([1.0, 3.0], dtype=torch.float32).to(device)  # 2类：背景, 目标
        label_smoothing = 0.3 # 标签平滑
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    elif args.loss == 'BCDE':
        criterion = BceDiceLoss(wb=1, wd=1)
    else:
        raise ValueError(f"未知损失函数: {args.loss}")

    # ========== 训练 ==========
    best_mean_iou = 0.0
    train_loss_list, val_loss_list = [], []
    train_miou_list, val_miou_list = [], []

    # ========== 断点续训部分 ==========
    start_epoch = 0
    best_mean_iou = 0.0
    train_loss_list, val_loss_list = [], []
    train_miou_list, val_miou_list = [], []

    if args.train:
        # ==== 加载checkpoint（如果存在） ====
        if os.path.exists(checkpoint_path):
            print(f"检测到 checkpoint，恢复断点训练：{checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_mean_iou = checkpoint.get('best_mean_iou', 0.0)
            start_epoch = checkpoint.get('epoch', 0)
            print(f"恢复epoch={start_epoch}，best_mean_iou={best_mean_iou}")
        else:
            print("未检测到 checkpoint，从头训练...")

        val_freq = 5  # 每隔 5 轮才验证一次
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            validate = ((epoch + 1) % val_freq == 0 or epoch == args.epochs - 1)

            train_loss, val_loss, lr, train_miou, val_miou, val_confmat = train_one_epoch(
                model=model,
                optim=optimizer,
                train_loader=trainLoader,
                test_loader=valLoader,
                device=device,
                loss_fuc=criterion,
                num_classes=num_classes,
                validate=validate
            )
            
            scheduler.step()

            # ... 训练完、scheduler.step() 后
            if validate and val_miou is not None:
                if val_miou > best_mean_iou:
                    best_mean_iou = val_miou
                    torch.save(model.state_dict(), weights_path)
                with open(log_path, "a") as f:
                    f.write(f"[epoch: {epoch+1}]\n{val_confmat}\n\n")

            # 安全打印
            vloss_str = f"{val_loss:.4f}" if val_loss is not None else "—"
            vmiou_str = f"{val_miou:.4f}" if val_miou is not None else "—"
            print(f"[epoch {epoch+1}/{args.epochs}] "
                f"train_loss={train_loss:.4f}, train_miou={train_miou:.4f}, "
                f"val_loss={vloss_str} val_miou={vmiou_str} lr={lr:.6f} "
                f"time={time.time()-epoch_start:.2f}s")

            train_loss_list.append(train_loss)
            train_miou_list.append(train_miou)
            val_loss_list.append(val_loss if val_loss is not None else float('nan'))
            val_miou_list.append(val_miou if val_miou is not None else float('nan'))

            # ====== 关键：每轮都保存checkpoint ======
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mean_iou': best_mean_iou,
            }
            torch.save(checkpoint, checkpoint_path)
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2
                mem_reserved = torch.cuda.memory_reserved() / 1024 ** 2
                print(f"  当前显存分配: {mem_allocated:.2f} MB, 保留: {mem_reserved:.2f} MB")
                max_mem_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
                max_mem_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
                print(f"  最大分配: {max_mem_allocated:.2f} MB, 最大保留: {max_mem_reserved:.2f} MB")
            print("[epoch:%d]" % (epoch+1))
            print("learning rate:%.8f" % lr)
            print("train loss:%.4f \t train mean iou:%.4f" % (train_loss, train_miou))
            vloss = val_loss if isinstance(val_loss, (int, float)) else float('nan')
            vmiou = val_miou if isinstance(val_miou, (int, float)) else float('nan')
            print("val   loss:%.4f \t val   mean iou:%.4f" % (vloss, vmiou), end='\n\n')
            now = datetime.datetime.now()
            print("Current time:", now.strftime('%Y-%m-%d %H:%M:%S'), end='\n\n')

        # 训练完成后绘图
        plt_loss_iou(train_loss_list, val_loss_list, train_miou_list, val_miou_list, iou_path)
        print(f"[OK] 训练曲线已保存：{iou_path}")

    # ========== 测试与逐图像导出 ==========
    # 只要权重存在即可
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"未找到最佳权重：{weights_path}。如果你设置了 --train False，请先确保存在 best_model.pth。")

    if args.network == 'DeepLab':
        with torch.inference_mode():
            model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    per_image_csv = os.path.join(per_image_dir, f"{args.network}_per_image.csv")
    test_miou, test_confmat, per_image = test_with_per_image(
        model=model,
        test_loader=testLoader,
        device=device,
        num_classes=num_classes,
        save_csv_path=per_image_csv,
        ignore_index=255,
    )

    print("test mean iou: %.4f" % (test_miou), end='\n\n')

    log_dict = test_confmat.to_dict()

    print(log_dict,type(log_dict))
    print(summary(model, input_size=(1, 3, 224, 224)))
    print(f"[OK] per-image CSV 导出：{per_image_csv}")

    # 保存 test 日志并汇总到 CSV
    model_name = args.network
    with open(log_path, "a") as f:
        info = f"[model: {model_name}]\n" + str(test_confmat) + '\n\n'
        f.write(info)
        append_log_to_excel(log_dict, model_name, categories, excel_path)

    try:
        torch.save(model, full_model_file)
        print(f"[OK] Full model saved to {full_model_file}")
    except :
        print(f"[Warning] Could not save full model due to pickling error")
        print("Falling back to saving only state_dict...")


    



# =============== 启动入口 ===============
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="segmentation training & evaluation")
    parser.add_argument('--train', default=True, type=bool, help="True: 训练+测试；False: 仅加载 best_model 测试")
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument('--network', default='DenseUNet', type=str)  
    parser.add_argument('--input-size-h', default=224, type=int)
    parser.add_argument('--input-size-w', default=224, type=int)
    parser.add_argument('--input-channels', default=3, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--loss', default='CE', type=str, choices=['CE', 'BCDE'])
    parser.add_argument('--opt', default='RMSprop', type=str)
    parser.add_argument('--data', default='Fruit_9', type=str)
    parser.add_argument('--sch', default='ReduceLROnPlateau', type=str)
    #parser.add_argument('--txt-path', default='./data/grayList.txt', type=str)
    parser.add_argument('--save-dir', default=None, type=str, help="save-data")
    args = parser.parse_args()
    print(args)
    main(args)
