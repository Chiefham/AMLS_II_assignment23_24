import pandas as pd

def cal_class_weight(train_csv_path,label_name='label'):

    train_df = pd.read_csv(train_csv_path)
    class_counts = train_df[label_name].value_counts().sort_index()

    # 计算每个类别的权重
    total_samples = train_df.shape[0]
    class_weights = {}
    for class_label, count in class_counts.items():
        class_weights[class_label] = total_samples / (len(class_counts) * count)

    return class_weights


