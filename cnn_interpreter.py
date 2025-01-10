import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import time

# Specify the classes for MNIST
classes = tuple(str(i) for i in range(10))

class CustomImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        # Return image and its filename
        return image, self.image_files[idx]


########################## Useful Functions ########################### 
def train(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if i % 100 == 99:
            print(f'[{i + 1}] loss: {running_loss / 100:.3f} | acc: {100.*correct/total:.2f}%')
            running_loss = 0.0
    
    return 100. * correct / total

def evaluate(model, testloader, criterion, device, dataset_name="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    class_correct = [0]*10
    class_total = [0]*10

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    accuracy = 100. * correct / total
    avg_loss = test_loss / len(testloader)
    print(f'{dataset_name} set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    print('\nPer-class accuracy:')
    for i in range(10):
        if class_total[i] > 0:
            class_accuracy = 100. * class_correct[i] / class_total[i]
            print(f'{classes[i]:>5s}: {class_accuracy:.2f}%')

    return accuracy, avg_loss

def evaluate_custom(model, dataloader, criterion, device, dataset_name="Custom Dataset"):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for batch_idx, (images, filepaths) in enumerate(dataloader):
            images = images.to(device)

            # Retrieve true labels for the current batch
            labels = torch.tensor(
                [int(filepath.split('_')[1].split('(')[0]) for filepath in filepaths],
                device=device,
            )
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1

    accuracy = 100. * correct / total
    avg_loss = test_loss / len(dataloader)
    print(f'{dataset_name} set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    print('\nPer-class accuracy:')
    for i in range(10):
        if class_total[i] > 0:
            class_accuracy = 100. * class_correct[i] / class_total[i]
            print(f'{classes[i]:>5s}: {class_accuracy:.2f}%')

    return accuracy, avg_loss

####################### Define the CNN architecture #######################
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Modified for MNIST (1 input channel instead of 3)
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        # Modified for MNIST's 28x28 input size
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

############################## Main Function ###############################
def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ######################### Data Preprocessing #########################
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    ############################# Datasets #############################
    # MNIST training dataset
    trainset = torchvision.datasets.MNIST(root='./dataset', train=True,
                                        download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128,
                            shuffle=True, num_workers=2)
    
    # MNIST test dataset
    testset = torchvision.datasets.MNIST(root='./dataset', train=False,
                                        download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100,
                        shuffle=False, num_workers=2)
    
    # Custom dataset from folder
    custom_dataset = CustomImageFolder('./generated_numbers', transform=transform)
    custom_loader = DataLoader(custom_dataset, batch_size=1,
                             shuffle=False, num_workers=2)

    ######################### Model Training ##########################
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    num_epochs = 1  # Reduced epochs as MNIST is easier to learn
    best_acc = 0
    train_acc_history = []
    test_acc_history = []

    print(f"Training on {device}")
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        train_acc = train(model, trainloader, criterion, optimizer, device)
        test_acc, test_loss = evaluate(model, testloader, criterion, device)
        
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        
        scheduler.step(test_loss)
        
        if test_acc > best_acc:
            print(f'Saving best model with accuracy: {test_acc:.2f}%')
            best_acc = test_acc

    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time/60:.2f} minutes')
    
    # Final evaluation on both test sets
    print("\nFinal Evaluation:")
    mnist_acc, _ = evaluate(model, testloader, criterion, device, "MNIST Test")
    custom_acc, _ = evaluate_custom(model, custom_loader, criterion, device, "Custom Generated")

if __name__ == '__main__':
    main()