import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.DenseNet import DenseNet  # 改为DenseNet
from models.get_models.DenseUNet_MDAF_EFA import DenseUNet_MDAF_EFA
# from model import SwinTransformerSys as SwinUnet
from torchvision.transforms import ToPILImage
melon_colors = {
        0: [0, 0, 0],        # Black - 黑色背景，对比度最高
        1: [255, 165, 0],    # Orange - 橙色代表甜瓜果实，更接近真实颜色
        2: [0, 255, 0],      # Green - 绿色代表茎部，符合植物特征
    }
def resize_image_to_prediction(original_img, prediction):
    # 调整原始图像的尺寸以匹配预测结果的尺寸
    original_img_resized = original_img.resize((prediction.shape[1], prediction.shape[0]), Image.BICUBIC)
    # 将预测结果转换为彩色图像
    original_img_color = colorize_prediction(prediction, melon_colors)
    return original_img_resized, original_img_color

def blend_images(original_img, prediction_color):
    # 确保 original_img 是一个 PIL 图像
    if not isinstance(original_img, Image.Image):
        original_img = Image.fromarray(original_img)

    # 确保 prediction_color 是一个 PIL 图像
    if not isinstance(prediction_color, Image.Image):
        prediction_color = Image.fromarray(prediction_color)

    # 调整预测图像的尺寸以匹配原始图像的尺寸
    prediction_color_resized = prediction_color.resize(original_img.size, Image.BICUBIC)

    # 将两个图像融合，这里使用简单的叠加方式
    blended_img = Image.blend(original_img.convert('RGBA'), prediction_color_resized.convert('RGBA'), alpha=0.5).convert('RGB')

    return blended_img

def colorize_prediction(prediction, color_map):
    # 初始化一个全黑的图像
    colored_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    
    # 遍历color_map中的每个颜色映射
    for idx, color in color_map.items():
        # 如果idx不为0，则将预测结果中等于idx的像素设置为对应的颜色
        colored_img[prediction == idx] = color
    
    return Image.fromarray(colored_img)
def main():

    weights_path = "./run_results/2025-10-14-04-48_DenseUNet_MDAF_EFA_Data_Fruit_Seg/best_model.pth"
    test_path = 'inference/input'
    
    # 甜瓜图片分类数已在训练时确定，无需读取grayList.txt
    num_classes = 3  # 甜瓜分割的类别数

    # get devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model - 使用DenseNet模型
    model = DenseUNet_MDAF_EFA(n_classes=num_classes)

    model.load_state_dict(torch.load(weights_path))
    model.to(device)

    model.eval()  # 进入验证模式

    # inference 所有图片路径
    test_imgs = [os.path.join(test_path, i) for i in os.listdir(test_path)]

    # load image
    for test_img in test_imgs:
        # print(test_img)
        original_img = Image.open(test_img).convert('RGB')
        data_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        img = data_transform(original_img)
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            output = model(img.to(device))

            prediction = output.argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)

            colored_prediction = colorize_prediction(prediction, melon_colors)

            # 调整原始图像尺寸并转换为彩色图像
            original_img_resized, original_img_color = resize_image_to_prediction(original_img, prediction)
            blended_img = blend_images(original_img_resized, original_img_color)
            # 将原始图像与预测的彩色图像融合
            blended_img = blend_images(original_img_resized, colored_prediction)

            # 保存融合后的图像
            a = test_img.split('.')[-2]
            save_path = a + '.png'
            img_path = save_path.replace('input', 'output')
            blended_img.save(img_path)  # 保存融合后的图像

if __name__ == '__main__':
    main()
