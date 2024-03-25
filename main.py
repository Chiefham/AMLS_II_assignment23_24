from train_test_split import train_test_split


def main():
    images_path = './Datasets/images/'
    label_path = './Datasets/labels.csv'

    # 首先划分训练集测试集
    tar_train_dir = './NewDatasets/train'
    tar_test_dir = './NewDatasets/test'
    test_ratio = 0.1
    train_test_split(images_path,tar_train_dir,tar_test_dir,test_ratio)


if __name__ == "__main__":
    main()