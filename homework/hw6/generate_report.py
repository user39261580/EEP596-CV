import torch
import torchvision
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import os

# Import from Assignment6
from Assignment6 import (
    compute_num_parameters,
    GAPNet,
    TransferFromResNet18Model,
    MobileNetV1,
    backbone,
    device
)

def generate_html_report():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Assignment 6 Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }
        .task {
            margin-bottom: 40px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .output {
            background-color: #ffffff;
            border: 1px solid #ddd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        .code {
            background-color: #f4f4f4;
            padding: 10px;
            border-left: 3px solid #3498db;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }
        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <h1>EEP 596 Computer Vision - Assignment 6 Report</h1>
    <p><strong>Date:</strong> November 8, 2025</p>
"""

    # Task 1: Number of Parameters
    html_content += """
    <div class="task">
        <h2>Task 1: Number of Parameters</h2>
        <p><strong>Objective:</strong> Write a function to compute the number of trainable parameters in a model (e.g., ResNet-34).</p>
        
        <div class="code">
def compute_num_parameters(net:nn.Module):
    num_para = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return num_para
        </div>
        
        <p><strong>Test with ResNet-34:</strong></p>
        <div class="output">
"""
    
    try:
        resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        num_params = compute_num_parameters(resnet34)
        html_content += f"<p>Number of trainable parameters in ResNet-34: <strong>{num_params:,}</strong></p>\n"
    except Exception as e:
        html_content += f"<p>Error: {str(e)}</p>\n"
    
    html_content += """
        </div>
    </div>
"""

    # Task 2: Global Average Pooling
    html_content += """
    <div class="task">
        <h2>Task 2: Global Average Pooling (GAPNet)</h2>
        <p><strong>Objective:</strong> Create and train GAPNet for 10 epochs on CIFAR-10 dataset.</p>
        
        <p><strong>Network Architecture:</strong></p>
        <div class="code">
GAPNet(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0)
  (conv2): Conv2d(6, 10, kernel_size=(5, 5), stride=(1, 1))
  (gap): AvgPool2d(kernel_size=10, stride=10, padding=0)
  (fc): Linear(in_features=10, out_features=10, bias=True)
)
        </div>
        
        <p><strong>Training Parameters:</strong></p>
        <ul>
            <li>Epochs: 10</li>
            <li>Learning Rate: 0.001</li>
            <li>Momentum: 0.9</li>
            <li>Optimizer: SGD</li>
        </ul>
        
        <p><strong>Training Output:</strong></p>
        <div class="output">
        <pre>
"""
    
    # Read training output for GAPNet
    try:
        with open('./report/Training_output_Gap_net_10epoch_gpu.txt', 'r') as f:
            gap_training_output = f.read()
        html_content += gap_training_output
    except Exception as e:
        html_content += f"Error reading training output: {str(e)}"
    
    html_content += """
        </pre>
        </div>
        
        <p><strong>Model Information:</strong></p>
        <div class="output">
"""
    
    try:
        gap_model = GAPNet()
        num_params_gap = compute_num_parameters(gap_model)
        html_content += f"<p>Number of trainable parameters in GAPNet: <strong>{num_params_gap:,}</strong></p>\n"
        html_content += f"<p>Model saved as: <strong>Gap_net_10epoch.pth</strong></p>\n"
    except Exception as e:
        html_content += f"<p>Error: {str(e)}</p>\n"
    
    html_content += """
        </div>
    </div>
"""

    # Task 3: Backbones
    html_content += """
    <div class="task">
        <h2>Task 3: Backbones (Feature Extraction)</h2>
        <p><strong>Objective:</strong> Download ResNet-18 (pretrained on ImageNet), remove the final fully connected layer, and extract features for "cat_eye.jpg".</p>
        
        <div class="code">
def backbone():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove final FC layer
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open('cat_eye.jpg').convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        features = model(image_tensor)
    
    return features
        </div>
        
        <p><strong>Feature Extraction Output:</strong></p>
        <div class="output">
"""
    
    try:
        # Check if cat_eye.jpg exists, if not create a placeholder message
        if os.path.exists('cat_eye.jpg'):
            features = backbone()
            html_content += f"<p>Extracted features shape: <strong>{features.shape}</strong></p>\n"
            html_content += f"<p>Feature vector dimensions: <strong>[batch_size={features.shape[0]}, features={features.shape[1]}, height={features.shape[2]}, width={features.shape[3]}]</strong></p>\n"
            html_content += f"<p>Total feature elements: <strong>{features.numel():,}</strong></p>\n"
        else:
            html_content += "<p><em>Note: cat_eye.jpg not found in current directory. Feature extraction requires this image file.</em></p>\n"
    except Exception as e:
        html_content += f"<p>Error during feature extraction: {str(e)}</p>\n"
    
    html_content += """
        </div>
    </div>
"""

    # Task 4: Transfer Learning
    html_content += """
    <div class="task">
        <h2>Task 4: Transfer Learning with ResNet-18</h2>
        <p><strong>Objective:</strong> Use pretrained ResNet-18, modify the last layer for 10 classes (CIFAR-10), freeze all weights except the last layer, and train for 10 epochs.</p>
        
        <p><strong>Network Modifications:</strong></p>
        <div class="code">
class TransferFromResNet18Model(nn.Module):
    def __init__(self, num_classes=10):
        super(TransferFromResNet18Model, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Freeze all layers
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Replace final FC layer for CIFAR-10 (10 classes)
        resnet.fc = nn.Linear(512, num_classes)
        self.model = resnet
        </div>
        
        <p><strong>Training Parameters:</strong></p>
        <ul>
            <li>Epochs: 10</li>
            <li>Learning Rate: 0.001</li>
            <li>Momentum: 0.9</li>
            <li>Optimizer: SGD (only last layer)</li>
            <li>Batch Size: 32</li>
        </ul>
        
        <p><strong>Training Output:</strong></p>
        <div class="output">
        <pre>
"""
    
    # Read training output for ResNet
    try:
        with open('./report/Training_output_Res_net_10epoch.txt', 'r') as f:
            resnet_training_output = f.read()
        html_content += resnet_training_output
    except Exception as e:
        html_content += f"Error reading training output: {str(e)}"
    
    html_content += """
        </pre>
        </div>
        
        <p><strong>Model Information:</strong></p>
        <div class="output">
"""
    
    try:
        transfer_model = TransferFromResNet18Model(num_classes=10)
        # Count only trainable parameters (should be just the final layer)
        trainable_params = sum(p.numel() for p in transfer_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in transfer_model.parameters())
        html_content += f"<p>Total parameters in model: <strong>{total_params:,}</strong></p>\n"
        html_content += f"<p>Trainable parameters (final layer only): <strong>{trainable_params:,}</strong></p>\n"
        html_content += f"<p>Frozen parameters: <strong>{total_params - trainable_params:,}</strong></p>\n"
        html_content += f"<p>Model saved as: <strong>Res_net_10epoch.pth</strong></p>\n"
    except Exception as e:
        html_content += f"<p>Error: {str(e)}</p>\n"
    
    html_content += """
        </div>
    </div>
"""

    # Task 5: MobileNet
    html_content += """
    <div class="task">
        <h2>Task 5: MobileNet Implementation</h2>
        <p><strong>Objective:</strong> Implement MobileNetV1 in PyTorch with depthwise separable convolutions, batch normalization, and ReLU activation.</p>
        
        <p><strong>Key Components:</strong></p>
        <ul>
            <li>Standard Convolution + BatchNorm + ReLU</li>
            <li>Depthwise Separable Convolution (Depthwise + Pointwise)</li>
            <li>Global Average Pooling</li>
            <li>Fully Connected Classifier</li>
        </ul>
        
        <div class="code">
class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()
        
        def conv_bn(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                         stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def conv_dw(in_channels, out_channels, stride):
            return nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                         stride=stride, padding=1, groups=in_channels, 
                         bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                # Pointwise Convolution
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.features = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(1024, n_classes)
        </div>
        
        <p><strong>Dimension Check:</strong></p>
        <div class="output">
"""
    
    try:
        ch_in = 3
        n_classes = 1000
        model = MobileNetV1(ch_in=ch_in, n_classes=n_classes)
        
        # Create random input
        x = torch.randn(1, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        html_content += f"<p>Input shape: <strong>{list(x.shape)}</strong></p>\n"
        html_content += f"<p>Output shape: <strong>{list(output.shape)}</strong></p>\n"
        
        # Verify dimensions
        if output.shape == (1, n_classes):
            html_content += f"<p style='color: green;'><strong>✓ Dimension check PASSED!</strong></p>\n"
        else:
            html_content += f"<p style='color: red;'><strong>✗ Dimension check FAILED!</strong> Expected (1, {n_classes}), got {output.shape}</p>\n"
        
        # Count parameters
        num_params_mobile = compute_num_parameters(model)
        html_content += f"<p>Number of trainable parameters in MobileNetV1: <strong>{num_params_mobile:,}</strong></p>\n"
        
    except Exception as e:
        html_content += f"<p>Error during dimension check: {str(e)}</p>\n"
    
    html_content += """
        </div>
    </div>
"""

    # Summary Table
    html_content += """
    <div class="task">
        <h2>Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Task</th>
                    <th>Model/Component</th>
                    <th>Status</th>
                    <th>Output File</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Task 1</td>
                    <td>Parameter Counter (ResNet-34)</td>
                    <td>✓ Completed</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Task 2</td>
                    <td>GAPNet Training</td>
                    <td>✓ Completed</td>
                    <td>Gap_net_10epoch.pth</td>
                </tr>
                <tr>
                    <td>Task 3</td>
                    <td>ResNet-18 Feature Extraction</td>
                    <td>✓ Completed</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Task 4</td>
                    <td>Transfer Learning (ResNet-18)</td>
                    <td>✓ Completed</td>
                    <td>Res_net_10epoch.pth</td>
                </tr>
                <tr>
                    <td>Task 5</td>
                    <td>MobileNetV1 Implementation</td>
                    <td>✓ Completed</td>
                    <td>-</td>
                </tr>
            </tbody>
        </table>
    </div>
"""

    # Close HTML
    html_content += """
</body>
</html>
"""

    # Save the HTML report
    report_path = './report/assignment6_report.html'
    os.makedirs('./report', exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Report generated successfully: {report_path}")
    return report_path


if __name__ == '__main__':
    generate_html_report()
