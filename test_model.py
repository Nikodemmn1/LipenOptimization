import glob
import os
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassF1Score
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from SchoolEqDataset import SchoolEqDataset
from SchoolEqModel import SchoolEqModel
from argparse import ArgumentParser
import tracemalloc
import time
import sys
from torchsummary import summary


def main(args):
    tracemalloc.start()
    print("Num GPUs Available: ", torch.cuda.device_count())

    # Info for the confusion matrix:
    num_classes = args.num_classes
    class_names = args.class_names.split(',')

    # CONFIGURATION VARIABLES
    training_epochs = 10
    test_batch_size = 256
    imgs_info_csv_path = args.labels
    model_weights_path = args.weights
    output_base_path = os.path.splitext(model_weights_path)[0]
    log_file = open(output_base_path + ".txt", "w")

    # DATALOADER, MODEL, LOSS FUNCTION
    imgs_test_paths = glob.glob("./dataset/test/**/*.jpg", recursive=True)
    test_dataset = SchoolEqDataset(imgs_test_paths, imgs_info_csv_path)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    model = SchoolEqModel(num_classes).cuda()

    loss_function = nn.CrossEntropyLoss(reduction='mean')

    # LOAD WEIGHTS
    if model_weights_path is not None:
        model_weights = torch.load(model_weights_path)
        # Remove a head if needed:
        if not args.quantization and model_weights[list(model_weights.keys())[-2]].shape[0] != num_classes:
            if args.test:
                print("Error Loaded model has wrong number of classes")
                return -1
            del model_weights[list(model_weights.keys())[-1]]  # bias
            del model_weights[list(model_weights.keys())[-1]]  # weights
        model.load_state_dict(model_weights, strict=False)
        print(f"Weights loaded successfully, from: {model_weights_path}")

    model.eval()

    # QUANTIZATION PREPARATION
    if args.quantization:
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
        pseudo_train(model, test_dataloader, loss_function, 1, log_file, use_gpu=True)
        model.eval()
        print("Quantization preparations complete.")

    print("Neutral and inference tests:")
    log_file.write("Neutral and inference tests:\n")
    if args.quantization:
        evaluate(model, test_dataloader, loss_function, num_classes, class_names, True, output_base_path,
                 log_file, False)
    else:
        evaluate(model, test_dataloader, loss_function, num_classes, class_names, False, output_base_path,
                 log_file, True)
        evaluate(model, test_dataloader, loss_function, num_classes, class_names, False, output_base_path,
                 log_file, False, print_summary_and_file_size=False)

    print("\n\nTraining tests:")
    log_file.write("\n\nTraining tests:\n")
    pseudo_train(model, test_dataloader, loss_function, training_epochs, log_file, use_gpu=True)
    pseudo_train(model, test_dataloader, loss_function, training_epochs, log_file, use_gpu=False)

    log_file.close()
    print("Tests complete.")


def pseudo_train(model, dataloader, loss_function, training_epochs, log_file, use_gpu, printout=True):
    if use_gpu:
        model = model.cuda()
    else:
        model = model.cpu()

    optimizer = Adam(model.parameters(), lr=1e-20)

    training_epoch_times = []
    tracemalloc.reset_peak()
    torch.cuda.reset_peak_memory_stats()

    model.train()
    for epoch in range(0, training_epochs):
        start_time_tmp = time.time()
        for x, y_true in tqdm(dataloader):
            if use_gpu:
                x = (x.float() / 255).cuda()
                y_true = y_true.cuda()
            else:
                x = (x.float() / 255).cpu()
                y_true = y_true.cpu()

            y = model(x)
            loss = loss_function(y, y_true)
            loss.backward()
            optimizer.step()
        training_epoch_times.append(time.time() - start_time_tmp)

    _, memory_peak = tracemalloc.get_traced_memory()
    memory_peak /= 1e6
    gpu_memory_peak = torch.cuda.max_memory_allocated() / 1e6
    average_t_epoch_time = sum(training_epoch_times) / len(training_epoch_times)

    if printout:
        results_string = ("[Training]"
                          f"[GPU {str(use_gpu)}]Average training epoch time: {average_t_epoch_time:.4f} s"
                          f" - Training memory peak: {memory_peak:.3f} MB"
                          f" - Training GPU memory peak: {gpu_memory_peak:.3f} MB")
        print(results_string)
        log_file.write(results_string + "\n")


def evaluate(model, dataloader, loss_function, num_classes, class_names, quantization, output_base_path, log_file,
             use_gpu, print_summary_and_file_size=True):
    if use_gpu:
        model = model.cuda()
    else:
        model = model.cpu()

    if quantization:
        model = torch.ao.quantization.convert(model.eval(), inplace=False).cpu()

    if print_summary_and_file_size:
        if not quantization:
            summary(model, (1, 224, 224))

            # Summary to file
            orig_stdout = sys.stdout
            sys.stdout = log_file
            summary(model, (1, 224, 224))
            sys.stdout = orig_stdout

        size_mb = get_mb_size_of_model(model)
        size_mb_string = f'\nModel file size (MB): {size_mb}\n'
        print(size_mb_string)
        log_file.write(size_mb_string + "\n")

    if use_gpu:
        conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes).cuda()
        accuracy = MulticlassAccuracy(num_classes, average='micro').cuda()
        f1_score = MulticlassF1Score(num_classes, average='macro').cuda()
    else:
        conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes).cpu()
        accuracy = MulticlassAccuracy(num_classes, average='micro').cpu()
        f1_score = MulticlassF1Score(num_classes, average='macro').cpu()

    running_loss = 0
    batches_cnt = 0
    inference_times = []

    tracemalloc.reset_peak()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for x, y_true in tqdm(dataloader):
            if use_gpu:
                x = (x.float() / 255).cuda()
                y_true = y_true.cuda()
            else:
                x = (x.float() / 255).cpu()
                y_true = y_true.cpu()
            start_time_tmp = time.time()
            y = model(x)
            inference_times.append(time.time() - start_time_tmp)

            loss = loss_function(y, y_true)
            running_loss += loss.item()
            batches_cnt += 1

            accuracy.update(y, y_true)
            f1_score.update(y, y_true)
            conf_matrix.update(y, y_true)
    _, memory_peak = tracemalloc.get_traced_memory()
    memory_peak /= 1e6
    gpu_memory_peak = torch.cuda.max_memory_allocated() / 1e6
    average_inference_time = sum(inference_times) / len(inference_times)

    loss = running_loss / batches_cnt
    accuracy_value = accuracy.compute()
    f1_score_value = f1_score.compute()
    conf_matrix_fig, _ = conf_matrix.plot(labels=class_names)

    results_string = ("[Inference]"
                      f"[GPU {str(use_gpu)}]Loss: {loss} - Accuracy: {accuracy_value} - F1 Score: {f1_score_value}"
                      f" - Average inference time: {average_inference_time:.4f} s"
                      f" - Memory peak: {memory_peak:.3f} MB - GPU memory peak: {gpu_memory_peak:.3f} MB")
    print(results_string)
    log_file.write(results_string + "\n")

    conf_matrix_fig.savefig(output_base_path + ".png", bbox_inches='tight')
    plt.close(conf_matrix_fig)


def get_mb_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size_in_mb = os.path.getsize("temp.p")/1e6
    os.remove('temp.p')
    return size_in_mb


def make_parser():
    parser = ArgumentParser(description="Arguments for model testing")
    parser.add_argument('--labels', '-l', type=str, default="./dataset/ReducedDatasetLabel.csv",
                        help='path to labels file .csv')
    parser.add_argument('--weights', '-w', type=str, required=False,
                        help='path to pretrained weights .pt')
    parser.add_argument("--quantization", action="store_true", help="use if tested model was subjected to quantization")
    parser.add_argument('--num-classes', type=int, default=3, help="Number of classes in the dataset",
                        choices=[3, 6])
    parser.add_argument('--class-names', type=str, default="WritingTool, Rubber, MeasurementTool", required=False,
                        choices=["WritingTool, Rubber, MeasurementTool",
                                 "Pen, Pencil, Rubber, Ruler, Triangle, None"],
                        help="Names of the classes in the dataset, presented in format A, B, C, ...")
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    main(args)

