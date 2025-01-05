import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

class MLP(nn.Module):
    def __init__(self, num_classes=10, input_size=3072):
        super(MLP, self).__init__()
        # Input size: 32x32x3 = 3072 (flattened RGB image)
        self.flatten = nn.Flatten()
        
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, 2048), # 3072 -> 2048
            nn.BatchNorm1d(2048), # Batch normalization
            nn.ReLU(), # Activation function
            nn.Dropout(0.3), # Dropout layer
            
            # Output layer
            nn.Linear(2048, num_classes)
        )
        
        # Class labels
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')
        
        # Empty arrays to store the training history
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x
    
    def prepare_dataset(self):
        """Initialize datasets and create dataloaders"""

        # Training transformations with data augmentation
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Test transformations
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Download and prepare datasets
        self.train_dataset = torchvision.datasets.CIFAR10(
            root='./dataset', 
            train=True,
            download=True, 
            transform=self.transform_train
        )
        self.test_dataset = torchvision.datasets.CIFAR10(
            root='./dataset', 
            train=False,
            download=True, 
            transform=self.transform_test
        )

        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=128,
            shuffle=True, 
            num_workers=2 
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=256,
            shuffle=False,
            num_workers=2
        )
    
    def plot_metrics(self):
        """Plot training and testing metrics"""
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.test_accuracies, label='Testing Accuracy')
        plt.title('Training vs Testing Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def evaluate_model(self, test_loader, loss_function, device):
        """Evaluate the model's loss and accuracy on the test set"""
        
        # Set model to evaluation mode
        self.eval()  

        # Initialize variables to track loss and accuracy
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device) # Move to device
                outputs = self(images) # Forward pass
                loss = loss_function(outputs, labels) # Calculate loss
                
                test_loss += loss.item() 
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def train_model(self, epochs, learning_rate):
        """Training process"""

        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move model to computing device
        self.to(device)
        print(f"Training on {device}")

        # Initialize dataset and dataloaders
        self.prepare_dataset()
        
        # Loss function
        loss_function = nn.MultiMarginLoss() # Hinge loss for multi-class classification
        
        # Optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )

        best_accuracy = 0
        start_time = time.time()

        # Training loop
        for epoch in range(epochs):
            # Set model to training mode
            self.train()

            # Train 
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):  # Added batch_idx for progress tracking
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad() # Reset the gradients of model parameters
                outputs = self(images) # Forward pass
                loss = loss_function(outputs, labels) # Calculate loss
                loss.backward() # Backward pass
                optimizer.step() # Update weights
                
                # Track training statistics
                running_loss += loss.item() # Add loss to running loss
                _, predicted = outputs.max(1) # Get the class index with the highest probability
                total += labels.size(0) # Add batch size to total
                correct += predicted.eq(labels).sum().item() # Add number of correct predictions to correct
                
                # Print progress every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    print(f'Batch [{batch_idx + 1}/{len(self.train_loader)}]')
            
            # Calculate training metrics
            train_loss = running_loss / len(self.train_loader)
            train_accuracy = 100. * correct / total
            
            # Each epoch evaluate the model on the test set
            test_loss, test_accuracy = self.evaluate_model(self.test_loader, loss_function, device)
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            self.test_accuracies.append(test_accuracy)
            
            # Print progress
            print(f'Epoch [{epoch + 1}/{epochs}]')
            print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%')
            print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(test_loss)
            
            # Save best model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                print(f'New best accuracy: {best_accuracy:.2f}%')
                torch.save(self.state_dict(), 'best_mlp_model.pth')

        # Training summary
        training_time = time.time() - start_time
        print(f'\nTraining completed in {training_time/60:.2f} minutes')
        print(f'Best accuracy: {best_accuracy:.2f}%')       

def main():
    # Initialize model
    model = MLP()
    # Train model
    model.train_model(epochs=30, learning_rate=0.01)
    # Plot training history
    model.plot_metrics()

if __name__ == '__main__':
    main()