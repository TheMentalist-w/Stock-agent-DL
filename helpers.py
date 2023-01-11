from matplotlib import pyplot as plt


def plot_history(history):

    plt.cla()
    plt.figure(figsize=(12, 7), dpi=80)

    plt.title("Train and test loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")

    epochs = range(1, history.params["epochs"] + 1)
    plt.plot(epochs, history.history["loss"], label="train_loss")
    plt.plot(epochs, history.history["val_loss"], label="test_loss")
    plt.legend()

    plt.savefig("train_test_loss.png")