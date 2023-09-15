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


def main():
    print("Num GPUs Available: ", torch.cuda.device_count())

    learning_rate = 3e-4
    batch_size = 256
    val_batch_size = 256
    val_every_n_epochs = 5
    max_epochs = 5000
    imgs_info_csv_path = "./dataset/UniformDatasetLabel.csv"

    random_seed = 175801
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    training_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    os.mkdir(f"./trained_models/{training_timestamp}")
    
    # DATA LOADING

    imgs_train_paths = glob.glob("./dataset/train/**/*.jpg", recursive=True)
    imgs_val_paths = glob.glob("./dataset/eval/**/*.jpg", recursive=True)

    train_dataset = SchoolEqDataset(imgs_train_paths, imgs_info_csv_path)
    val_dataset = SchoolEqDataset(imgs_val_paths, imgs_info_csv_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

    # AUGMENTATION IMAGE TRANSFORMS

    img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomApply(nn.ModuleList([
            torchvision.transforms.RandomAffine(degrees=30, translate=(0.05, 0.05), shear=10),
        ]), p=0.2),
        torchvision.transforms.RandomApply(nn.ModuleList([
            torchvision.transforms.GaussianBlur(kernel_size=5),
        ]), p=0.24),
        torchvision.transforms.RandomApply(nn.ModuleList([
            torchvision.transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(1, 1.5), saturation=(0.5, 1.5),
                                               hue=(-0.1, 0.1)),
        ]), p=0.2),
        torchvision.transforms.RandomPosterize(4, 0.1),
        torchvision.transforms.RandomHorizontalFlip(0.2),
        torchvision.transforms.RandomVerticalFlip(0.2),
    ])

    # MODEL, OPTIMIZER, LOSS FUNCTION, ETC.

    model = SchoolEqModel().cuda()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # optimizer = SGD(model.parameters(), lr=learning_rate)
    writer = SummaryWriter()
    loss_function = nn.CrossEntropyLoss(reduction='mean')

    # RigL

    total_iterations = len(train_dataloader) * max_epochs
    t_end = int(0.75 * total_iterations)
    pruner = RigLScheduler(model, optimizer, T_end=t_end, dense_allocation=0.2)

    # TRAINING LOOP

    for epoch in range(0, max_epochs):

        # TRAINING PART

        train_running_loss = 0
        train_batches = 0
        train_samples = 0
        train_correct = 0

        conf_matrix = MulticlassConfusionMatrix(num_classes=6).cuda()

        model.train(True)
        for x, y_true in tqdm(train_dataloader):
            x = img_transforms(x)
            x = (x.float() / 255).cuda()  # normalizing to 0-1 and moving to GPU
            y_true = y_true.cuda()
            optimizer.zero_grad()

            y = model(x)

            loss = loss_function(y, y_true)
            loss.backward()

            if pruner():
                optimizer.step()

            train_running_loss += loss.item()
            _, y_class = torch.max(y, dim=1)
            train_correct += (y_class == y_true).sum().item()

            train_batches += 1
            train_samples += len(y_true)

            conf_matrix.update(y, y_true)

        train_loss = train_running_loss / train_batches
        train_accuracy = train_correct / train_samples

        print(f"Epoch {epoch} - train loss: {train_loss} - train accuracy: {train_accuracy}")

        conf_matrix_fig, _ = conf_matrix.plot(labels=["pen", "pencil", "rubber",
                                                      "ruler", "triangle", "none"])

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_figure("Confusion Matrix/train", conf_matrix_fig, epoch)

        plt.close(conf_matrix_fig)

        # VALIDATION PART

        if epoch % val_every_n_epochs == 0:
            val_running_loss = 0
            val_batches = 0
            val_samples = 0
            val_correct = 0

            conf_matrix = MulticlassConfusionMatrix(num_classes=6).cuda()

            model.eval()
            with torch.no_grad():
                for x, y_true in tqdm(val_dataloader):
                    x = (x.float() / 255).cuda()
                    y_true = y_true.cuda()
                    y = model(x)
                    loss = loss_function(y, y_true)

                    val_running_loss += loss.item()
                    _, y_class = torch.max(y, dim=1)
                    val_correct += (y_class == y_true).sum().item()

                    val_batches += 1
                    val_samples += len(y_true)

                    conf_matrix.update(y, y_true)

            val_loss = val_running_loss / val_batches
            val_accuracy = val_correct / val_samples

            conf_matrix_fig, _ = conf_matrix.plot(labels=["pen", "pencil", "rubber",
                                                          "ruler", "triangle", "none"])

            print(f"Epoch {epoch} - val loss: {val_loss} - val accuracy: {val_accuracy}")

            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_accuracy, epoch)
            writer.add_figure("Confusion Matrix/val", conf_matrix_fig, epoch)

            plt.close(conf_matrix_fig)

        # LAST PART - SAVING MODEL AND EPOCH RESULTS

        writer.flush()
        torch.save(model.state_dict(), f'./trained_models/{training_timestamp}/{epoch}.pt')

    writer.close()


if __name__ == '__main__':
    main()
