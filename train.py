from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
from public import MyDataset, DEVICE, CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 训练参数
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_WORKERS = 0

train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.GaussNoise(p=0.2),
        A.Rotate(limit=25, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

valid_transform = A.Compose(
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


def train_model(model, num_epochs, train_dataloader, valid_dataloader, criterion, optimizer, device, scheduler):
    writer = SummaryWriter('logs')

    # 轮次迭代
    for epoch in tqdm(range(num_epochs)):  # 显示迭代进度
        batch_loss_list = []
        model.train()  # 切换到训练模式

        # 训练
        for img, mask in tqdm(train_dataloader):
            # 取出一batch数据
            img = img.float().to(device)
            mask = mask.float().to(device)
            # 训练
            prediction = model(img)  # 前向预测
            loss = criterion(prediction[0], mask)  # 计算损失
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            # 记录日志
            batch_loss_list.append(loss.item())

        # 在验证集中测试损失和准确率
        model.eval()  # 切换到推断模式
        with torch.no_grad():  # 不记录梯度
            train_loss = sum(batch_loss_list) / len(batch_loss_list)
            valid_loss, valid_accuracy = valid_model(model, valid_dataloader, criterion, device=device)

        # 输出信息
        print(f'Epoch: {epoch + 1:02d}/{num_epochs:02d} '
              f'| Train Loss: {train_loss :.4f} '
              f'| Validation Loss: {valid_loss :.4f} '
              f'| Validation Accuracy: {valid_accuracy * 100 :.1f}%')
        # 在TensorBoard中显示
        writer.add_scalar('Train Loss', train_loss, epoch + 1)
        writer.add_scalar('Validation Loss', valid_loss, epoch + 1)
        writer.add_scalar('Validation Accuracy', valid_accuracy, epoch + 1)

        # 利用验证损失更新学习率
        scheduler.step(valid_loss)

    writer.close()


def valid_model(model, valid_dataloader, criterion, device):
    accuracy = 0
    loss = 0
    with torch.no_grad():
        # 迭代验证集batch
        for img, mask in valid_dataloader:
            img = img.float().to(device)
            mask = mask.float().to(device)
            # 预测
            prediction = model(img)
            batch_loss = criterion(prediction[0], mask)
            loss += batch_loss.item()

            tp, fp, fn, tn = smp.metrics.get_stats(prediction[0], mask.int(), threshold=0.5,
                                                   mode='binary')  # 真阳性、假阳性、假阴性、真阴性
            accuracy_batch = smp.metrics.accuracy(tp, fp, fn, tn, reduction='micro').item()  # 像素准确率
            accuracy += accuracy_batch
    loss /= len(valid_dataloader)
    accuracy /= len(valid_dataloader)
    return loss, accuracy  # 验证损失和准确率


if __name__ == '__main__':
    # 随机种子
    random_seed = 717
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  # CPU
    torch.cuda.manual_seed(random_seed)  # GPU
    os.environ['PYTHONHASHSEED'] = str(random_seed)  # 为了禁止hash随机化，使得实验可复现
    # torch.backends.cudnn.deterministic = True  # 强制确定加速算法
    # torch.backends.cudnn.benchmark = False  # 禁止寻找最优算法

    # 训练集目录
    data_dir = './data/train'
    img_dir = os.path.join(data_dir, 'image_mha')
    mask_dir = os.path.join(data_dir, 'label_mha')

    # 数据路径列表
    img_path = sorted([os.path.join(img_dir, file_name) for file_name in os.listdir(img_dir)])
    mask_path = sorted([os.path.join(mask_dir, file_name) for file_name in os.listdir(mask_dir)])

    # 验证集划分
    validation = 0.2  # 验证集比例
    img_path_train, img_path_val, mask_path_train, mask_path_val = train_test_split(img_path, mask_path,
                                                                                    test_size=validation,
                                                                                    random_state=random_seed)

    # dataset
    train_dataset = MyDataset(img_path_train, mask_path_train, transform=train_transform)
    valid_dataset = MyDataset(img_path_val, mask_path_val, transform=valid_transform)

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    # 创建UNet模型
    ENCODER = 'mit_b0'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d'
    AUX_PARAMS = dict(
        pooling='max',
        dropout=0.2,
        activation='softmax',
        classes=len(CLASSES),
    )
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        decoder_attention_type=None,
        classes=len(CLASSES),
        activation=ACTIVATION,
        aux_params=AUX_PARAMS
    )
    model.to(DEVICE)  # 模型加到GPU中
    preprocessing = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # 训练设置
    criterion = SoftBCEWithLogitsLoss()  # 二元交叉熵损失函数，输入未经sigmoid函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 优化器
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min',
                                  verbose=True)  # 学习率调度器，这个是根据模型在验证集上的表现自动调整学习率

    # 训练
    train_model(model, NUM_EPOCHS, train_dataloader, valid_dataloader, criterion, optimizer, DEVICE, scheduler)

    # 保存模型
    model_dir = './model/'
    model_name = f'unet_mit_b0_{random_seed}.pt'
    torch.save(model, model_dir + model_name)
