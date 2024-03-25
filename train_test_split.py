import os
import shutil
import random



def train_test_split(source_dir,tar_train_dir,tar_test_dir,test_ratio):
    os.makedirs(tar_train_dir, exist_ok=True)
    os.makedirs(tar_test_dir, exist_ok=True)

    # obtain images list
    image_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    random.shuffle(image_files)

    # number for split
    num_total = len(image_files)
    num_test = int(num_total * test_ratio)

    # split
    test_files = image_files[:num_test]
    train_files = image_files[num_test:]

    # copy to train and test folders
    for filename in train_files:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(tar_train_dir, filename))
    for filename in test_files:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(tar_test_dir, filename))






