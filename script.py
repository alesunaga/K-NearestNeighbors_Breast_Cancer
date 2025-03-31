import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np  # Added for numerical operations

def evaluate_knn(data, target, test_size=0.2, random_state=100, max_k=100):
    """
    Evaluates the KNN classifier for different values of k and plots the results.

    Args:
        data: Features data.
        target: Target labels.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Seed for the random number generator.
        max_k: Maximum value of k to evaluate.

    Returns:
        None (displays a plot).
    """

    # Split data
    training_data, validation_data, training_labels, validation_labels = train_test_split(
        data, target, test_size=test_size, random_state=random_state
    )

    accuracies = []
    k_values = range(1, max_k + 1)  # Include max_k

    for k in k_values:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(training_data, training_labels)
        accuracies.append(classifier.score(validation_data, validation_labels))

    # Plotting
    plt.figure(figsize=(10, 6))  # Adjust figure size
    plt.plot(k_values, accuracies, marker='o', linestyle='-')  # Add markers
    plt.xlabel('K')
    plt.ylabel('Validation Accuracy')
    plt.title('Breast Cancer Classifier Accuracy vs. K')
    plt.xticks(np.arange(1, max_k + 1, 5)) # Improve x-axis ticks
    plt.grid(True)  # Add gridlines
    plt.tight_layout() # Improve layout
    plt.show()

# Load and evaluate
breast_cancer_data = load_breast_cancer()
evaluate_knn(breast_cancer_data.data, breast_cancer_data.target)

#Example of using a different test size
#evaluate_knn(breast_cancer_data.data, breast_cancer_data.target, test_size=0.3)
