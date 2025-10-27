import torch
import torchvision
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Runs on {device} device.")


def CIFAR10_dataset_a():
    """write the code to grab a single mini-batch of 4 images from the training set, at random. 
   Return:
    1. A batch of images as a torch array with type torch.FloatTensor. 
    The first dimension of the array should be batch dimension, the second channel dimension, 
    followed by image height and image width. 
    2. Labels of the images in a torch array

    """
    # Convert dataset to tensor and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./cifar10', 
        train=True,
        download=True, 
        transform=transform
    )
    
    # Initialize DataLoader with batch_size=4 and shuffle
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=4,
        shuffle=True
    )
    
    # Get first batch
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 
    #            'dog', 'frog', 'horse', 'ship', 'truck')
    # fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    # for i in range(4):
    #     img = images[i] / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     axes[i].imshow(np.transpose(npimg, (1, 2, 0)))
    #     axes[i].text(0, -2, classes[labels[i]], fontsize=12)
    #     axes[i].axis('off')
    # plt.show()

    return images, labels

class Net(nn.Module):
    # Use this function to define your network
    # Creates the network
    def __init__(self):
        super().__init__()
        # 3 channels to 6 channels, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 2x2 max pooling
        self.pool = nn.MaxPool2d(2, 2)
        # 6 channels to 16 channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Full connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16 channels, 5x5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 output classes

    def forward(self, x):
        # First conv -> ReLU -> 池化
        x = self.pool(F.relu(self.conv1(x)))
        # Second conv -> ReLU -> 池化
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten
        x = torch.flatten(x, 1)
        # Full connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_classifier():
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
    net = Net()
    net.to(device) # Move to GPU if available
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    # Training for 2 epochs
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # Move to GPU if available
            
            optimizer.zero_grad() # Reset gradients due to accumulation in PyTorch implementation
            
            # forward + backward + optimize
            outputs = net(inputs)
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
    PATH = './cifar_net_2epoch_gpu.pth'
    torch.save(net.state_dict(), PATH)

def evalNetwork():
    # Initialize the network and load trained weights
    PATH = './cifar_net_2epoch.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH, map_location=device)) # Load weights to the appropriate device
    net.to(device) # Move to GPU if available
    net.eval()  #  Set to evaluation mode

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
            outputs = net(images)
            # Get predicted results
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on 10000 test images: {accuracy:.2f}%')
    return accuracy

def convert_cuda_weights_to_CPU(gpu_weights_path):
    model = Net()
    cpu_device = torch.device('cpu')
    state_dict = torch.load(gpu_weights_path, map_location=cpu_device)

    # load the weights into the model
    model.load_state_dict(state_dict)
    model.to(cpu_device)
    print("Model weights loaded to CPU.")

    cpu_weights_path = './cifar_net_2epoch.pth'
    torch.save(model.state_dict(), cpu_weights_path)
    print(f"CPU-version model weights saved to: {cpu_weights_path}")

def get_first_layer_weights():
    net = Net()
    PATH = './cifar_net_2epoch.pth'
    net.load_state_dict(torch.load(PATH)) # Load the trained weights
    first_weight = net.conv1.weight.data.clone()  # Get conv1 weights (exclude bias)

    return first_weight

def get_second_layer_weights():
    net = Net()
    PATH = './cifar_net_2epoch.pth'
    net.load_state_dict(torch.load(PATH)) # Load the trained weights
    second_weight = net.conv2.weight.data.clone()  # Get conv2 weights (exclude bias)
    return second_weight

# def hyperparameter_sweep():
#     '''
#     Reuse the CNN and training code from Question 2
#     Train the network three times using different learning rates: 0.01, 0.001, and 0.0001
#     During training, record the training loss every 2000 iterations
#     compute and record the training and test errors every 2000 iterations by randomly sampling 1000 images from each dataset
#     After training, plot three curves
#     '''
#     return None

if __name__ == "__main__":
    # images, labels = CIFAR10_dataset_a()
    # train_classifier()
    # evalNetwork()
    # convert_cuda_weights_to_CPU('./cifar_net_2epoch_gpu.pth')
    weight1 = get_first_layer_weights()
    weight2 = get_second_layer_weights()