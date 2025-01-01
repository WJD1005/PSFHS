from matplotlib import pyplot as plt
import random
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from public import MyDataset, DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH

test_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)


def plot_test_demo(model, test_dataset, index, index_number, device):
    img = test_dataset[index][0].permute(1, 2, 0)  # 转成显示形式
    mask = test_dataset[index][1].permute(1, 2, 0)

    # 预测
    input_img = test_dataset[index][0].unsqueeze(0).float().to(device)  # 多套一层batch维度
    model.eval()
    with torch.no_grad():
        prediction = model(input_img)
    prediction = prediction[0].squeeze(0).cpu().permute(1, 2, 0)  # 去掉batch维度，转存到cpu中等下转numpy，顺便转成显示形式
    prediction[prediction < 0.5] = 0  # 二值处理再画图
    prediction[prediction >= 0.5] = 1

    plt.figure()
    plt.suptitle(f'NO.{index_number:05d}')
    # Ground Truth
    plt.subplot(2, 4, 1)
    plt.imshow(img)  # 原图
    plt.title('RAW Image')
    plt.subplot(2, 4, 2)
    plt.imshow(mask[:, :, 0])  # 背景
    plt.title('Ground Truth Background')
    plt.subplot(2, 4, 3)
    plt.imshow(mask[:, :, 1])  # 耻骨
    plt.title('Ground Truth PS')
    plt.subplot(2, 4, 4)
    plt.imshow(mask[:, :, 2])  # 胎头
    plt.title('Ground Truth FH')

    # 预测
    plt.subplot(2, 4, 5)
    plt.imshow(img)  # 原图
    plt.title('RAW Image')
    plt.subplot(2, 4, 6)
    plt.imshow(prediction[:, :, 0])  # 背景
    plt.title('Prediction Background')
    plt.subplot(2, 4, 7)
    plt.imshow(prediction[:, :, 1])  # 耻骨
    plt.title('Prediction PS')
    plt.subplot(2, 4, 8)
    plt.imshow(prediction[:, :, 2])  # 胎头
    plt.title('Prediction FH')
    plt.show()


if __name__ == '__main__':
    # 测试集目录
    data_dir = './data/test'
    img_dir = os.path.join(data_dir, 'image_mha')
    mask_dir = os.path.join(data_dir, 'label_mha')

    # 数据路径列表
    img_path = sorted([os.path.join(img_dir, file_name) for file_name in os.listdir(img_dir)])
    mask_path = sorted([os.path.join(mask_dir, file_name) for file_name in os.listdir(mask_dir)])

    # dataset
    test_dataset = MyDataset(img_path, mask_path, transform=test_transform)

    # 加载模型
    model_dir = './model/'
    model_name = 'unet_mit_b0_717.pt'
    model = torch.load(model_dir + model_name)
    model.to(DEVICE)

    # 随机挑一个测试例
    random_index = random.randint(0, len(test_dataset) - 1)
    index_number = int(img_path[random_index][-9:-4])  # 把真实编号切出来
    plot_test_demo(model, test_dataset, random_index, index_number, DEVICE)
