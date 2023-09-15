import datetime
import glob
import os
import random
import numpy as np
from torchmetrics.classification import MulticlassConfusionMatrix
import torchvision.transforms.v2
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from SchoolEqDataset import SchoolEqDataset
from SchoolEqModel import SchoolEqModel
from rigl_torch.RigL import RigLScheduler


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def test_model(model, test_dataloader, loss_function, print_results=True):
    test_running_loss = 0
    test_batches = 0
    test_samples = 0
    test_correct = 0

    with torch.no_grad():
        for x, y_true in tqdm(test_dataloader):
            x = x.float() / 255
            y_true = y_true
            y = model(x)
            loss = loss_function(y, y_true)

            test_running_loss += loss.item()
            _, y_class = torch.max(y, dim=1)
            test_correct += (y_class == y_true).sum().item()

            test_batches += 1
            test_samples += len(y_true)

    if print_results:
        test_loss = test_running_loss / test_batches
        test_accuracy = test_correct / test_samples
        print(f"Test loss: {test_loss} - test accuracy: {test_accuracy}")


def main():
    num_classes = 3
    model = SchoolEqModel(num_classes).cuda()
    test_batch_size = 32

    # LOAD WEIGHTS
    model_weights = torch.load("trained_models/20230915181321/best_430.pt")
    # Remove a head if needed:
    if model_weights[list(model_weights.keys())[-2]].shape[0] != num_classes:
        del model_weights[list(model_weights.keys())[-1]]  # bias
        del model_weights[list(model_weights.keys())[-1]]  # weights
    model.load_state_dict(model_weights, strict=False)
    model = model.cpu()

    imgs_test_paths = glob.glob("./dataset/test/**/*.jpg", recursive=True)
    imgs_info_csv_path = "./dataset/ReducedDatasetLabel.csv"
    test_dataset = SchoolEqDataset(imgs_test_paths, imgs_info_csv_path)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    loss_function = nn.CrossEntropyLoss(reduction='mean')

    model.eval()
    print_size_of_model(model)

    test_model(model, test_dataloader, loss_function)

    torch.ao.quantization.fuse_modules(model, inplace=True, modules_to_fuse=[
        ["feature_extractor.0", "feature_extractor.1", "feature_extractor.2"],
        ["feature_extractor.4", "feature_extractor.5", "feature_extractor.6"],
        ["feature_extractor.8", "feature_extractor.9", "feature_extractor.10"],
        ["feature_extractor.12", "feature_extractor.13", "feature_extractor.14"],
        ["feature_extractor.16", "feature_extractor.17", "feature_extractor.18"]
    ])

    model.qconfig = torch.ao.quantization.default_qconfig
    print(model.qconfig)
    torch.ao.quantization.prepare(model, inplace=True)
    
    test_model(model, test_dataloader, loss_function, print_results=False)

    torch.ao.quantization.convert(model, inplace=True)

    test_model(model, test_dataloader, loss_function)

    print_size_of_model(model)


if __name__ == '__main__':
    main()
