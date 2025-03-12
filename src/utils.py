import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss(history):
    """Plot training and validation loss."""
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_reconstruction_errors(errors, threshold):
    """Plot reconstruction errors for anomaly detection."""
    plt.hist(errors, bins=50, label='Reconstruction Errors')
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2, label='Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
