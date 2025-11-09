import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from torchvision import models
from PIL import Image

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Runs on {device} device.")



def compute_num_parameters(net:nn.Module):
    """compute the number of trainable parameters in *net* e.g., ResNet-34.  
    Return the estimated number of parameters Q1. 
    """
    num_para = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print (f'The number of trainable parameters: {num_para}')

    return num_para


def CIFAR10_dataset_a():

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    return images, labels


class GAPNet(nn.Module):
    def __init__(self):
        super(GAPNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.gap = nn.AvgPool2d(10) 
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.gap(F.relu(self.conv2(x)))
        x = x.view(-1, 10) # Flatten
        x = self.fc(x)
        return x



def train_GAPNet():
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./cifar10', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )
    
    # Initialize the network
    model = GAPNet()
    model.to(device) # Move to GPU if available
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Training for 10 epochs
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # Move to GPU if available
            
            optimizer.zero_grad() # Reset gradients due to accumulation in PyTorch implementation
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Print progress
            running_loss += loss.item()
            if i % 2000 == 1999:  # Update every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    
    print('Finished Training')
    
    # Save the trained model weights
    PATH = './Gap_net_10epoch_gpu.pth'
    torch.save(model.state_dict(), PATH)


def eval_GAPNet():
    # Initialize the network and load trained weights
    PATH = './Gap_net_10epoch_gpu.pth'
    model = GAPNet()
    model.load_state_dict(torch.load(PATH, map_location=device)) # Load weights to the appropriate device
    model.eval()  #  Set to evaluation mode
    model.to(device) # Move to GPU if available

    # Load the test dataset
    batch_size = 4
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./cifar10', train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Compute accuracy
    correct = 0
    total = 0
    
    with torch.no_grad():  # No need to compute gradients during evaluation
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # Move to GPU if available
            outputs = model(images)
            # Get predicted results
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on 10000 test images: {accuracy:.2f}%')
    return accuracy

def backbone():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    model = nn.Sequential(*list(model.children())[:-1]) # Remove the final fully connected layer
    model.eval() # Set to evaluation mode

    # Define transform for ResNet
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open('cat_eye.jpg').convert('RGB')
    image_tensor = transform(image).unsqueeze(0) # Add batch dimension
    with torch.no_grad():
        features = model(image_tensor)
        # print(f'Extracted features shape: {features.shape}')

    return features


class TransferFromResNet18Model(nn.Module):
    def __init__(self, num_classes=10):
        super(TransferFromResNet18Model, self).__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Freeze all layers
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer for CIFAR-10 (10 classes)
        # ResNet18's fc layer input features: 512
        resnet.fc = nn.Linear(512, num_classes)
        
        # Store the modified resnet as the model
        self.model = resnet
    
    def forward(self, x):
        return self.model(x)


def transfer_learning():
    model = TransferFromResNet18Model(num_classes=10)
    model = model.to(device)

    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    batch_size = 32
    trainset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    # Set up loss function and optimizer (only optimize the final fc layer)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.model.fc.parameters(), lr=0.001, momentum=0.9)
    
    # Training loop
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader, 0):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Print progress
            running_loss += loss.item()
            if i % 200 == 199:  # Update every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
    
    print('Finished Training')

    # Save the ResNet18 model (not the wrapper)
    torch.save(model.model.state_dict(), 'Res_net_10epoch_gpu.pth')
    print('Model saved as Res_net_10epoch_gpu.pth')

    # Evaluation
    testset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on 10000 test images: {accuracy:.2f}%')


# class MobileNetV1(nn.Module):
#     """Define MobileNetV1 please keep the strucutre of the class Q5"""
#     def __init__(self, ch_in, n_classes):


#     def forward(self, x):

def convert_cuda_weights_to_CPU(gpu_weights_path, cpu_weights_path):
    """Convert GPU model weights to CPU-compatible weights."""
    cpu_device = torch.device('cpu')
    
    # Load the state dict from GPU weights
    state_dict = torch.load(gpu_weights_path, map_location=cpu_device)
    
    print("Model weights loaded to CPU.")

    # Save the state dict for CPU
    torch.save(state_dict, cpu_weights_path)
    print(f"CPU-version model weights saved to: {cpu_weights_path}")

    
if __name__ == '__main__':
    #Q1
    # from torchvision import models
    # resnet34 = models.resnet34(pretrained=True)
    # num_para = compute_num_parameters(resnet34)
    
    # train_GAPNet()
    # eval_GAPNet()
    # convert_cuda_weights_to_CPU('./Gap_net_10epoch_gpu.pth', './Gap_net_10epoch.pth')
    # backbone()

    # Re-train and save the model with correct structure
    transfer_learning()
    convert_cuda_weights_to_CPU('./Res_net_10epoch_gpu.pth', './Res_net_10epoch.pth')

    # Q5
    # ch_in=3
    # n_classes=1000
    # model = MobileNetV1(ch_in=ch_in, n_classes=n_classes)
