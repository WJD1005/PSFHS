from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm
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


def eval_model(model, test_dataloader, device):
    sensitivity = 0
    specificity = 0
    accuracy = 0
    iou = 0
    f1_score = 0

    model.eval()  # 切换推断模式
    with torch.no_grad():
        for img, mask in tqdm(test_dataloader):
            img = img.float().to(device)
            mask = mask.float().to(device)
            # 预测
            prediction = model(img)

            # 计算指标
            tp, fp, fn, tn = smp.metrics.get_stats(prediction[0], mask.int(), threshold=0.5,
                                                   mode='binary')  # 真阳性、假阳性、假阴性、真阴性
            sensitivity_batch = smp.metrics.sensitivity(tp, fp, fn, tn, reduction="micro").item()  # TPR
            specificity_batch = smp.metrics.specificity(tp, fp, fn, tn, reduction="micro").item()  # TNR
            accuracy_batch = smp.metrics.accuracy(tp, fp, fn, tn, reduction='micro').item()  # 像素准确率
            iou_batch = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()  # 交并比
            f1_score_batch = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()  # F1
            sensitivity += sensitivity_batch
            specificity += specificity_batch
            accuracy += accuracy_batch
            iou += iou_batch
            f1_score += f1_score_batch

    sensitivity /= len(test_dataloader)
    specificity /= len(test_dataloader)
    accuracy /= len(test_dataloader)
    iou /= len(test_dataloader)
    f1_score /= len(test_dataloader)

    return sensitivity, specificity, accuracy, iou, f1_score


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

    # dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=2, num_workers=0, shuffle=False)

    # 加载模型
    model_dir = './model/'
    model_name = 'unet_mit_b0_717.pt'
    model = torch.load(model_dir + model_name)
    model.to(DEVICE)

    # 测试
    sensitivity, specificity, accuracy, iou, f1_score = eval_model(model, test_dataloader, DEVICE)

    print(f'Sensitivity: {sensitivity}')
    print(f'Specificity: {specificity}')
    print(f'Accuracy: {accuracy}')
    print(f'IoU: {iou}')
    print(f'F1-score: {f1_score}')
