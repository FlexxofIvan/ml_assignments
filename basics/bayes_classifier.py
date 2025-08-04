import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

def norm_distr(mu, sigma, x):

    z = (x - mu)
    return (1 / (2 * np.pi) ** 0.5) * (1 / sigma) * np.exp(-0.5 * (z / sigma) ** 2)


def distribution(series, bins_count):

    minimum = series.min()
    maximum = series.max()
    step = (maximum - minimum) / bins_count
    counts = []
    for i in range(bins_count):
        filt = series[(series >= minimum + i * step) & (series < minimum + (i + 1) * step)]
        counts.append(len(filt))
    bin_centers = minimum + np.arange(bins_count) * step + step / 2
    densities = np.array(counts) / (sum(counts) * step)
    mean = np.sum(densities * bin_centers) * step
    sigma = np.sum((bin_centers - mean) ** 2 * densities) * step
    return bin_centers, densities, mean, sigma


def compute_class_prob_densities(features, data, bins_count=10):

    class_prob_densities = []
    classes = data['target'].unique()
    for cls in classes:
        class_data = data[data['target'] == cls]
        dens_product = np.ones(len(features))
        for col in features.columns:
            x = features[col]
            class_x = class_data[col]
            _, _, mean, sigma = distribution(class_x, bins_count)
            dens = norm_distr(mean, np.sqrt(sigma), x)
            dens_product *= dens
        class_prob_densities.append(dens_product)
    return class_prob_densities


def classify_samples(class_prob_densities):

    prob_matrix = np.vstack(class_prob_densities).T
    predicted_classes = np.argmax(prob_matrix, axis=1)
    return predicted_classes


def plot_results(features, true_labels, predicted_labels):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Predicted classes")
    plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=predicted_labels, cmap='viridis', alpha=0.7)
    plt.xlabel(features.columns[0])
    plt.ylabel(features.columns[1])

    plt.subplot(1, 2, 2)
    plt.title("True classes")
    plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=true_labels, cmap='viridis', alpha=0.7)
    plt.xlabel(features.columns[0])
    plt.ylabel(features.columns[1])

    plt.tight_layout()
    plt.show()


def main():
    
    dataset = datasets.load_iris(as_frame=True)
    features = dataset.data
    targets = dataset.target
    data = pd.concat([features, targets.rename('target')], axis=1)

    class_prob_densities = compute_class_prob_densities(features, data)
    predicted = classify_samples(class_prob_densities)

    plot_results(features, targets, predicted)

if __name__ == "__main__":
    main()
