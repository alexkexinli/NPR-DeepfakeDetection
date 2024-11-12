import torch
import numpy as np
from torch.ao.quantization.fx import convert

from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader
import torchvision.transforms as transforms
from PIL import Image

def predict(model, imgs):
    crop_func = transforms.Lambda(lambda img: img)
    flip_func = transforms.Lambda(lambda img: img)
    rz_func = transforms.Resize((opt.loadSize, opt.loadSize))
    covert_tensor = transforms.Compose([
        rz_func,
        crop_func,
        flip_func,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imgs = [covert_tensor(img) for img in imgs]
    with torch.no_grad():
        y_pred = []
        for img in imgs:
            print(type(img),"img")
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())

    realcounter = 0
    for pred in y_pred:
        if pred <=0.5:
            realcounter+=1
    return realcounter >=8


def validate(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print(y_pred.shape,"y_pred",y_pred)
    print(y_true.shape, "y_true",y_true)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
