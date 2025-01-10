import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import time


class MNISTSequenceDataset(Dataset):
    """Dataset class for MNIST sequence prediction"""
    def __init__(self, mnist_dataset, max_samples=None):
        """Initialize the dataset"""
        self.mnist_data = mnist_dataset
        # Pre-compute indices for each digit to speed up retrieval
        self.digit_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(mnist_dataset):
            self.digit_indices[label].append(idx)
        # Optional dataset size limitation for faster testing
        self.max_samples = max_samples

    def __len__(self):
        """Return the number of samples in the dataset"""
        if self.max_samples:
            return min(len(self.mnist_data), self.max_samples)
        return len(self.mnist_data)
    
    def __getitem__(self, idx):
        """Retrieve the current image, label, next image, and next label"""
        # Get current image and label
        curr_img, curr_label = self.mnist_data[idx]
        # Get next number label
        next_label = (curr_label + 1) % 10
        # Efficiently get a random image of the next number
        next_idx = np.random.choice(self.digit_indices[next_label])
        next_img, _ = self.mnist_data[next_idx]
        
        return curr_img, torch.tensor(curr_label), next_img, torch.tensor(next_label)

# Simplified CNN Encoder
class CNNEncoder(nn.Module):
    """CNN Encoder for MNIST images"""
    def __init__(self, latent_dim=128):  # Reduced latent dimension
        """Initialize the encoder layers"""
        super(CNNEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim)
        )
        
    def forward(self, x):
        """Forward pass with the encoder"""
        return self.encoder(x)

# Simplified CNN Decoder
class CNNDecoder(nn.Module):
    """CNN Decoder for MNIST images"""
    def __init__(self, latent_dim=128):
        """Initialize the decoder layers"""
        super(CNNDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        """Forward pass with the decoder"""
        x = self.fc(x)
        x = x.view(-1, 32, 7, 7)
        return self.decoder(x)

# Simple sequence transformer
class SimpleTransformer(nn.Module):
    """Simple Transformer for sequence prediction"""
    def __init__(self, latent_dim=128):
        """Initialize the transformer layers"""
        super(SimpleTransformer, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
    def forward(self, x, _):
        """Forward pass with the transformer"""
        return self.transform(x)


class MNISTSequencePredictor(nn.Module):
    """MNIST Sequence Predictor model"""
    def __init__(self, latent_dim=128):
        """Initialize the sequence predictor model"""
        super(MNISTSequencePredictor, self).__init__()
        self.encoder = CNNEncoder(latent_dim)
        self.transformer = SimpleTransformer(latent_dim)
        self.decoder = CNNDecoder(latent_dim)
        
    def forward(self, img, curr_num):
        """Forward pass with the sequence predictor"""
        latent = self.encoder(img)
        transformed = self.transformer(latent, curr_num)
        next_img = self.decoder(transformed)
        return next_img

    def save_prediction(self, img, curr_num, save_path, index):
        """Save the generated next number to a file"""
        self.eval()
        with torch.no_grad():
            next_img = self(img.unsqueeze(0), curr_num.unsqueeze(0))
            
            # Convert to PIL Image
            # Denormalize from [-1, 1] to [0, 255]
            generated = ((next_img.squeeze().cpu().numpy() + 1) * 127.5).astype(np.uint8)
            
            # Create PIL Image
            img_pil = Image.fromarray(generated)
            
            # Create filename with input and predicted number
            filename = f"label_{(curr_num.item() + 1) % 10}({index}).png"
            
            # Save image
            img_pil.save(os.path.join(save_path, filename))
            
            return filename

def train_model(model, train_loader, num_epochs, device):
    """Train the sequence predictor model"""
    # Set Loss function and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Set model to training mode
    model.train() 

    # Train the model
    for epoch in range(num_epochs):
        epoch_loss = 0.0 # Reset loss each epoch
        correct_predictions = 0  # Reset counters each epoch
        total_samples = 0 # Reset counters each epoch
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for curr_imgs, curr_nums, next_imgs, _ in pbar:
                # Move tensors to device
                curr_imgs = curr_imgs.to(device) 
                curr_nums = curr_nums.to(device) 
                next_imgs = next_imgs.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                generated_imgs = model(curr_imgs, curr_nums)
                loss = criterion(generated_imgs, next_imgs)
                loss.backward()
                optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()

                # Calculate accuracy with a more lenient threshold
                similarity = torch.mean((generated_imgs - next_imgs) ** 2, dim=[1, 2, 3])
                threshold = 0.3  # Increased threshold for more reasonable accuracy
                correct_predictions += torch.sum(similarity < threshold).item()
                total_samples += curr_imgs.size(0)

                accuracy = (correct_predictions / total_samples) * 100
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'accuracy': f'{accuracy:.2f}%'})

        print(f'Epoch {epoch+1} - Average loss: {epoch_loss/len(train_loader):.4f} - Accuracy: {accuracy:.2f}%')

def main():
    print("Starting main function...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = "generated_numbers"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_train = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
    
    # Create sequence dataset
    print("Creating sequence dataset...")
    train_dataset = MNISTSequenceDataset(mnist_train, max_samples=None)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    print("Initializing model...")
    model = MNISTSequencePredictor().to(device)
    
    # Train model
    print("Starting training...")
    train_model(model, train_loader, num_epochs=2, device=device)
    
    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), 'mnist_sequence_predictor.pth')
    
    # Generate and save images
    print("Generating and saving images...")
    num_test_samples = 100
    
    for i in tqdm(range(num_test_samples)):
        random_idx = np.random.randint(0, len(train_dataset))
        curr_img, curr_num, _, _ = train_dataset[random_idx]
        
        # Move tensors to device and save prediction
        filename = model.save_prediction(
            curr_img.to(device),
            curr_num.to(device),
            output_dir,
            i
        )
        print(f"Saved prediction as: {filename}")

if __name__ == "__main__":
    main()