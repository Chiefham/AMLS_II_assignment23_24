from train_test_split import train_test_split
from Data_Loader import Data_Loader
from Models import Models
import os
import tensorflow as tf
from Model_Evaluation import Model_Evaluation


def main():
    images_path = './Datasets/images/'
    label_path = './Datasets/labels.csv'

    # 首先划分训练集测试集,如果没有划分，则新建文件夹进行划分
    if not os.path.exists(os.path.join(os.getcwd(),'NewDatasets')):
        tar_train_dir = './NewDatasets/train'
        tar_test_dir = './NewDatasets/test'
        test_ratio = 0.1
        train_test_split(images_path, tar_train_dir, tar_test_dir, test_ratio)

    train_path = './NewDatasets/train'
    test_path = './NewDatasets/test'
    train_generator,val_generator,test_generator=Data_Loader(train_path,test_path,
                                                             label_path,height=128,width=128)

    model = Models(train_generator,val_generator,test_generator)
    # 判断是否已经存在训练好的模型文件
    if not os.path.exists(os.path.join(os.getcwd(),'SamCNN.model')):
        model.SamCNN()


    # 模型评估
    test_label_path = './NewDatasets/test_labels.csv'
    model_path = './SamCNN.model'
    Model_Evaluation(test_generator,model_path,test_label_path)









if __name__ == "__main__":
    main()