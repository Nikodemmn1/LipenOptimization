import pandas as pd
import numpy as np

def main():
    np.random.seed(1525)

    input_labels_path = "dataset/UniformDatasetLabel.csv"
    output_labels_path = "dataset/ReducedDatasetLabel.csv"

    input_labels = pd.read_csv(input_labels_path,sep=";")

    input_labels = input_labels.drop('Sublabel',axis=1)
    input_labels = input_labels.drop('Extra', axis=1)
    input_labels = input_labels.drop('Author', axis=1)

    # Merge classes
    input_labels['Label'] = input_labels['Label'].replace(1, 0)
    input_labels['Label'] = input_labels['Label'].replace(2, 1)
    input_labels['Label'] = input_labels['Label'].replace(4, 3)
    input_labels['Label'] = input_labels['Label'].replace(3, 2)

    # Remove the none class:
    input_labels = input_labels[input_labels['Label'] != 5]

    # randomly select labels to lie
    treshold = 0.1 # 10 %
    rand_rows = np.where(np.random.random(input_labels.shape[0]) < treshold)[0]

    # half will be change into the one class and half into the other
    rand_rows_1 = rand_rows[:int(len(rand_rows)/2)]
    rand_rows_2 = rand_rows[int(len(rand_rows)/2):]

    # change up the labels
    input_labels.iloc[rand_rows_1, 1] = input_labels.iloc[rand_rows_1, 1] - 1
    input_labels.iloc[rand_rows_2, 1] = input_labels.iloc[rand_rows_2, 1] - 2
    # loop around
    input_labels['Label'] = input_labels['Label'].replace(-1, 2)
    input_labels['Label'] = input_labels['Label'].replace(-2, 1)

    # save to out file
    input_labels.to_csv(output_labels_path,sep=";",index=False)

    print("Done")

if __name__ == '__main__':
    main()
