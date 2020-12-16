'''
fang.zong
'''
import cv2
import torch
from torchvision import transforms
import numpy as np
import random
from torch.autograd import Variable
import config

from FeatherNet.models.FeatherNet import FeatherNetB
from FeatherNet.tools.dataset import PredictDataset


def load_model(param_path):

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    model = FeatherNetB()
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else "cpu")
    model.to(device)

    if torch.cuda.is_available():
        checkpoint = torch.load(param_path)
    else:
        checkpoint = torch.load(param_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def predict(model, image_list):

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else "cpu")

    model.eval()

    normalize = transforms.Normalize(mean=config.mean,
                                     std=config.std)
    img_size = config.image_size
    test_dataset = PredictDataset(
        images=image_list,
        transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(image_list), shuffle=True,
                                              num_workers=0, pin_memory=False, sampler=None)
    preds = []
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            input_var = Variable(input).float().to(device)
            output = model(input_var)

            soft_output = torch.softmax(output, dim=-1)
            _, predicted = torch.max(soft_output.data, 1)
            preds.append(predicted)
    pred_label = preds[0]
    # pred_label = int(pred_label * 90)

    return pred_label


def main():
    test_dir = config.test_path
    model_path = config.model_path
    import os
    img_path_list = os.listdir(test_dir)
    img_list = list()
    for img_path in img_path_list:
        img = cv2.imread(os.path.join(test_dir, img_path))
        img_list.append(img)
    model = load_model(model_path)
    preds = predict(model, img_list)
    print(preds)


if __name__ == 'main':
    main()



