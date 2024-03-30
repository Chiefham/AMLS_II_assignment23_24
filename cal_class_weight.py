from sklearn.utils import class_weight


def cal_class_weight(train_generator):
    classes_to_predict = [0, 1, 2, 3, 4]
    class_weights = class_weight.compute_class_weight(class_weight="balanced",
                                                      classes=classes_to_predict,
                                                      y=train_generator.labels)
    class_weights_dict = {i: class_weights[i] for i, label in
                          enumerate(classes_to_predict)}

    print(class_weights_dict)
