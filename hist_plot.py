import matplotlib.pyplot as plt


def hist_plot(hist, pic_name):
    plt.figure(figsize=(15, 5))
    plt.plot(hist.epoch, hist.history["categorical_accuracy"], '-o',
             label='Train Accuracy', color='#ff7f0e')
    plt.plot(hist.epoch, hist.history["val_categorical_accuracy"], '-o',
             label='Val Accuracy', color='#1f77b4')
    plt.xlabel('Epoch', size=14)
    plt.ylabel('Accuracy', size=14)
    plt.legend(loc=2)

    plt2 = plt.gca().twinx()
    plt2.plot(hist.epoch, hist.history['loss'], '-o',
              label='Train Loss', color='#2ca02c')
    plt2.plot(hist.epoch, hist.history['val_loss'], '-o',
              label='Val Loss', color='#d62728')
    plt.legend(loc=3)
    plt.ylabel('Loss', size=14)
    plt.title("Model Accuracy and loss")

    plt.savefig(pic_name)
