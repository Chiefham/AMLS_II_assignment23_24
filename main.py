from Models import Models
from Data_Loader import my_image_augmentation, make_train_gen, make_val_gen
from train_test_split import train_test_split
import pandas as pd
from Model_Evaluation import Model_Evaluation
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def main():
    # global setting
    batch_size = 8
    target_size_dim = 300
    epochs = 10
    train_efficientnet = 1
    train_vgg = 1
    train_resnet = 1

    # load data
    train_label_path = './Datasets/train.csv'
    train_img_path = './Datasets/train_images/'
    df = pd.read_csv(train_label_path)
    df['path'] = train_img_path + df['image_id']
    df['label'] = df['label'].astype('str')
    X_train, X_valid = train_test_split(df, test_size=0.1, random_state=42,
                                        shuffle=True)
    train_gen = make_train_gen(X_train, x_col='path', y_col='label',
                               batch_size=batch_size,
                               target_size_dim=target_size_dim)
    valid_gen = make_val_gen(X_valid, x_col='path', y_col='label',
                             batch_size=batch_size * 2,
                             target_size_dim=target_size_dim)

    MODEL = Models(target_size_dim, train_gen, valid_gen, epochs)
    if train_efficientnet == 1:
        MODEL.EfficientNetB3()
    if train_vgg == 1:
        MODEL.VGG19()
    if train_resnet == 1:
        MODEL.ResNet101V2()

    # model evaluation
    efficientnetb3_path = './EfficientNetB3.model'
    vgg19_path = './VGG19.model'
    resnet101v2 = './ResNet101V2.model'

    Model_Evaluation(efficientnetb3_path, valid_gen)
    Model_Evaluation(vgg19_path,valid_gen)
    Model_Evaluation(resnet101v2, valid_gen)


if __name__ == "__main__":
    main()
