import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

# Step 1: Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data / 255.0  # Normalize pixel values to [0, 1]
y = mnist.target.astype(int)  # Convert target to integers

# Step 2: Apply PCA
n_components = 50  # Number of principal components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Step 3: Choose a digit to generate from
input_digit_index = 0  # Change this index to select different digits
input_digit = X_pca[input_digit_index]

# Step 4: Generate a new digit by adding a small random perturbation
perturbation = np.random.normal(0, 0.01, size=input_digit.shape)
new_digit_pca = input_digit + perturbation

# Step 5: Inverse transform to get the new digit in the original space
new_digit = pca.inverse_transform(new_digit_pca)

# Step 6: Visualize the original and generated digits
plt.figure(figsize=(8, 4))

# Original digit
plt.subplot(1, 2, 1)
plt.title("Original Digit")
plt.imshow(X.iloc[input_digit_index].values.reshape(28, 28), cmap='gray')  # Use .iloc here
plt.axis('off')

# Generated digit
plt.subplot(1, 2, 2)
plt.title("Generated Digit")
plt.imshow(new_digit.reshape(28, 28), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Visualize the generated digit for various n_components values
n_components_values = [50, 100, 200, 300, 400, 500, 600, 700]
plt.figure(figsize=(10, 20))

for i, n in enumerate(n_components_values):
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    input_digit = X_pca[input_digit_index]
    perturbation = np.random.normal(0, 0.01, size=input_digit.shape)
    new_digit_pca = input_digit + perturbation
    new_digit = pca.inverse_transform(new_digit_pca)
    
    plt.subplot(len(n_components_values), 1, i + 1)
    plt.title(f"Generated Digit with {n} components")
    plt.imshow(new_digit.reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()