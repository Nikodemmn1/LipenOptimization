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
import socket
from argparse import ArgumentParser
from time import perf_counter, strftime
from torchsummary import summary


def main(args):
    # Measure time
    start_time = perf_counter()

    print("Num GPUs Available: ", torch.cuda.device_count())

    # Info for the confusion matrix:
    num_classes = args.num_classes
    class_names = args.class_names.split(',')

    # HYPERPARAMS:
    use_pruner = args.prune
    learning_rate = 3e-4
    batch_size = 256
    val_batch_size = 256
    val_every_n_epochs = 5
    test_batch_size = 256
    max_epochs = 500
    imgs_info_csv_path = args.labels
    model_weights_path = args.weights

    random_seed = args.seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    unique_name = create_experiment_name()
    training_timestamp = f"./{args.run_dir}/{unique_name}"
    log_name = f"./{args.save_dir}/{unique_name}"
    os.mkdir(training_timestamp)
    writer = SummaryWriter(log_dir=log_name)

    # DATA LOADING

    imgs_train_paths = glob.glob("./dataset/train/**/*.jpg", recursive=True)
    imgs_val_paths = glob.glob("./dataset/eval/**/*.jpg", recursive=True)
    imgs_test_paths = glob.glob("./dataset/test/**/*.jpg", recursive=True)

    train_dataset = SchoolEqDataset(imgs_train_paths, imgs_info_csv_path)
    val_dataset = SchoolEqDataset(imgs_val_paths, imgs_info_csv_path)
    test_dataset = SchoolEqDataset(imgs_test_paths, imgs_info_csv_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

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

    model = SchoolEqModel(num_classes).cuda()
    if args.weighted_loss:
        loss_function = nn.CrossEntropyLoss(weight=torch.tensor([1.0,2.0,1.0]).cuda(),  reduction='mean')
    else:
        loss_function = nn.CrossEntropyLoss(reduction='mean')

    # LOAD WEIGHTS
    if model_weights_path is not None:
        model_weights = torch.load(model_weights_path)
        # Remove a head if needed:
        if model_weights[list(model_weights.keys())[-2]].shape[0] != num_classes:
            if args.test:
                print("Error Loaded model has wrong number of classes")
                return -1
            del model_weights[list(model_weights.keys())[-1]]  # bias
            del model_weights[list(model_weights.keys())[-1]]  # weights
        model.load_state_dict(model_weights, strict=False)
        print(f"Weights loaded successfully, from: {model_weights_path}")

    # Print model details:
    summary(model, (1, 224, 224))

    # Model saving variables:
    best_model = None
    last_model = None

    # Freezing:
    if args.freeze:
        model_params_tmp = [layer for layer in model.parameters()]
        optimizer = Adam([
            {'params': [model_params_tmp[0], model_params_tmp[4]], 'lr': 1e-8},
            {'params': model_params_tmp[1:4] + model_params_tmp[5:]}
        ], lr=learning_rate)
        print("First two layers have smaller learning rate: 1e-8" )
    else:
        optimizer = Adam(model.parameters(), lr=learning_rate)

    # RigL
    total_iterations = len(train_dataloader) * max_epochs
    t_end = int(0.75 * total_iterations)
    pruner = RigLScheduler(model, optimizer, T_end=t_end, dense_allocation=0.2) if use_pruner else None

    # Quantization
    if args.quantization:
        model.eval()
        torch.ao.quantization.fuse_modules(model, inplace=True, modules_to_fuse=[
            ["feature_extractor.0", "feature_extractor.1", "feature_extractor.2"],
            ["feature_extractor.4", "feature_extractor.5", "feature_extractor.6"],
            ["feature_extractor.8", "feature_extractor.9", "feature_extractor.10"],
            ["feature_extractor.12", "feature_extractor.13", "feature_extractor.14"],
            ["feature_extractor.16", "feature_extractor.17", "feature_extractor.18"]
        ])
        model.train()
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        torch.ao.quantization.prepare_qat(model, inplace=True)

    # Test Validate:
    if args.test:
        validate(model, test_dataloader, loss_function, num_classes, class_names, writer, -1, args.quantization, 'test')
        # Measure time
        end_time = datetime.datetime.utcfromtimestamp(perf_counter() - start_time).strftime(
            "%H hours %M minutes %S seconds")
        print(f"Test concluded without a fuss. It took {end_time}. Have a nice day! 😄")
        return

    # TRAINING LOOP

    for epoch in range(0, max_epochs):

        # TRAINING PART

        train_running_loss = 0
        train_batches = 0
        train_samples = 0
        train_correct = 0

        conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes).cuda()

        model.train(True)
        for x, y_true in tqdm(train_dataloader):
            x = img_transforms(x)
            x = (x.float() / 255).cuda()  # normalizing to 0-1 and moving to GPU
            y_true = y_true.cuda()
            optimizer.zero_grad()

            y = model(x)

            loss = loss_function(y, y_true)
            loss.backward()

            if use_pruner:
                if pruner():
                    optimizer.step()
            else:
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

        conf_matrix_fig, _ = conf_matrix.plot(labels=class_names)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_figure("Confusion Matrix/train", conf_matrix_fig, epoch)

        plt.close(conf_matrix_fig)

        # VALIDATION PART

        if epoch % val_every_n_epochs == 0 or epoch == max_epochs - 1:

            val_accuracy = validate(model, val_dataloader, loss_function, num_classes, class_names, writer, epoch,
                                    args.quantization)

            # LAST PART - SAVING MODEL AND EPOCH RESULTS
            # Save a new best checkpoint:
            if best_model is None or best_model[1] < val_accuracy:
                best_path = f'{training_timestamp}/best_{epoch}.pt'
                torch.save(model.state_dict(), best_path)
                # Remove last best checkpoint:
                if best_model is not None:
                    try:
                        os.remove(best_model[0])
                    except:
                        print("Error Removing second best model")
                # Save new best info:
                best_model = (best_path, val_accuracy)
            # Save last checkpoint:
            last_path = f'{training_timestamp}/last_{epoch}.pt'
            torch.save(model.state_dict(), last_path)
            # Remove last checkpoint:
            if last_model is not None:
                try:
                    os.remove(last_model)
                except:
                    print("Error Removing second last model")
            last_model = last_path

        # Validate on test at the end:
        if epoch == max_epochs - 1:
            validate(model, test_dataloader, loss_function, num_classes, class_names, writer, epoch, args.quantization,
                     'test')

        # Quantization
        if args.quantization:
            if epoch > 0.375 * max_epochs:
                model.apply(torch.ao.quantization.disable_observer)

            if epoch > 0.25 * max_epochs:
                model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)

    writer.close()
    # Print model details:
    print(pruner)
    # Measure End time
    end_time = datetime.datetime.utcfromtimestamp(perf_counter() - start_time).strftime(
        "%H hours %M minutes %S seconds")
    print(f"Training concluded without a fuss. It took {end_time}. Have a nice day! 😄")


def validate(model, val_dataloader, loss_function, num_classes, class_names, writer, epoch, quantization, mode='val'):
    # Validate model on validation set
    if quantization:
        model_without_quantization = model
        model = torch.ao.quantization.convert(model.eval().cpu(), inplace=False).cpu()
        conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes).cpu()
    else:
        conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes).cuda()
    val_running_loss = 0
    val_batches = 0
    val_samples = 0
    val_correct = 0

    model.eval()
    with torch.no_grad():
        for x, y_true in tqdm(val_dataloader):
            # Prepare input
            if quantization:
                x = (x.float() / 255).cpu()
                y_true = y_true.cpu()
            else:
                x = (x.float() / 255).cuda()
                y_true = y_true.cuda()
            # Interfere
            y = model(x)
            # Calculate Loss
            loss = loss_function(y, y_true)
            val_running_loss += loss.item()
            # Get prediction
            _, y_class = torch.max(y, dim=1)
            # Sum correct classifications
            val_correct += (y_class == y_true).sum().item()
            val_batches += 1
            val_samples += len(y_true)
            # Pass info to the matrix
            conf_matrix.update(y, y_true)
    # Calculate Loss and Accuracy
    val_loss = val_running_loss / val_batches
    val_accuracy = val_correct / val_samples
    # Create matrix
    conf_matrix_fig, _ = conf_matrix.plot(labels=class_names)
    # Print Validation scores
    print(f"Epoch {epoch} - {mode} loss: {val_loss} - {mode} accuracy: {val_accuracy}")
    # Save Info in tensorboard
    writer.add_scalar(f"Loss/{mode}", val_loss, epoch)
    writer.add_scalar(f"Accuracy/{mode}", val_accuracy, epoch)
    writer.add_figure(f"Confusion Matrix/{mode}", conf_matrix_fig, epoch)

    plt.close(conf_matrix_fig)
    writer.flush()

    if quantization:
        model = model_without_quantization.cuda()

    return val_accuracy


def create_experiment_name():
    # Get the current date and time
    current_datetime = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    # Get the hostname of the computer
    hostname = socket.gethostname()
    # Combine the components to create the final string
    return f"{current_datetime}_{hostname}"


def make_parser():
    parser = ArgumentParser(description="Arguments for model training")
    parser.add_argument('--labels', '-l', type=str, default="./dataset/ReducedDatasetLabel.csv",
                        help='path to labels file .csv')
    parser.add_argument('--weights', '-w', type=str, required=False,
                        help='path to pretrained weights .pt')
    parser.add_argument('--run-dir', '-r', type=str, default='runs',
                        help='path to tensorboard log dir')
    parser.add_argument('--save-dir', '-s', type=str, default='trained_models',
                        help='path where to save model after training')
    parser.add_argument("--prune", action="store_true", help="Enable pruning")
    parser.add_argument("--freeze", action="store_true", help="Enable first 2 layers freeze")
    parser.add_argument("--quantization", action="store_true", help="Enable quantization and QAT")
    parser.add_argument("--weighted-loss", action="store_true", help="Use weighted loss function (needs to change weights in code)")
    parser.add_argument('--num-classes', type=int, default=3, help="Number of classes in the dataset", choices=[3, 6])
    parser.add_argument('--class-names', type=str, default="WritingTool, Rubber, MeasurementTool", required=False,
                        choices=["WritingTool, Rubber, MeasurementTool", "Pen, Pencil, Rubber, Ruler, Triangle, None"],
                        help="Names of the classes in the dataset, presented in format A, B, C, ...")
    parser.add_argument("--test", action="store_true", help="Validate on test dataset"),
    parser.add_argument('--seed', type=int, default=175809, help="Seed used to controll randomness",
                        choices=[175801, 175867, 351668])

    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    main(args)
