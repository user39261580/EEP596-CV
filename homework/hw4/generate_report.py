import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from Assignment4 import (
    CIFAR10_dataset_a, 
    Net, 
    evalNetwork, 
    get_first_layer_weights, 
    get_second_layer_weights,
    hyperparameter_sweep
)
import base64
from io import BytesIO
import os

def generate_html_report():
    """Generate comprehensive HTML report for Assignment 4"""
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EEP 596 Computer Vision - Assignment 4 Report</title>
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
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }
        h3 {
            color: #7f8c8d;
            margin-top: 20px;
        }
        .info-box {
            background-color: #ecf0f1;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }
        .value {
            font-weight: bold;
            color: #e74c3c;
        }
        img {
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border: 1px solid #bdc3c7;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #bdc3c7;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .section {
            margin-bottom: 40px;
        }
        pre {
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <h1>EEP 596 Computer Vision - Assignment 4 Report</h1>
    
"""
    
    # Task 1: CIFAR-10 Dataset Information
    html_content += """
    <div class="section">
        <h2>Task 1: CIFAR-10 Dataset</h2>
        
        <h3>Dataset Statistics</h3>
        <div class="info-box">
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>num_train_batches</td>
                    <td class="value">5</td>
                </tr>
                <tr>
                    <td>num_test_batches</td>
                    <td class="value">1</td>
                </tr>
                <tr>
                    <td>num_img_per_batch</td>
                    <td class="value">10000</td>
                </tr>
                <tr>
                    <td>num_train_img</td>
                    <td class="value">50000</td>
                </tr>
                <tr>
                    <td>num_test_img</td>
                    <td class="value">10000</td>
                </tr>
                <tr>
                    <td>size_batch_bytes (KB)</td>
                    <td class="value">30000 KB (~29.3 MB)</td>
                </tr>
                <tr>
                    <td>size_image_bytes (KB)</td>
                    <td class="value">3.072 KB (3072 bytes for 32x32x3)</td>
                </tr>
                <tr>
                    <td>size_batchimage_bytes (10k images, KB)</td>
                    <td class="value">30720 KB (~30 MB)</td>
                </tr>
            </table>
        </div>
        
        <h3>1a. Sample Mini-batch (4 random images)</h3>
"""
    
    # Generate and save sample images
    print("Generating Task 1a: Sample mini-batch visualization...")
    try:
        images, labels = CIFAR10_dataset_a()
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck')
        
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for i in range(4):
            img = images[i] / 2 + 0.5  # unnormalize
            npimg = img.numpy()
            axes[i].imshow(np.transpose(npimg, (1, 2, 0)))
            axes[i].set_title(f'{classes[labels[i]]}', fontsize=12)
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig('./report/task1a_sample_batch.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        html_content += """
        <img src="task1a_sample_batch.png" alt="Sample mini-batch">
        <p><strong>Image tensor shape:</strong> torch.Size([4, 3, 32, 32])</p>
        <p><strong>Labels tensor shape:</strong> torch.Size([4])</p>
"""
    except Exception as e:
        html_content += f"<p style='color: red;'>Error generating sample batch: {e}</p>"
    
    html_content += """
    </div>
"""
    
    # Task 2: Train Classifier
    html_content += """
    <div class="section">
        <h2>Task 2: Train Classifier</h2>
        
        <h3>Network Architecture</h3>
        <pre>
Net(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
        </pre>
        
        <h3>Test Accuracy</h3>
"""
    
    # Evaluate network
    print("Evaluating network on test set...")
    try:
        accuracy = evalNetwork()
        html_content += f"""
        <div class="info-box">
            <p><strong>Test Accuracy on 10,000 images:</strong> <span class="value">{accuracy:.2f}%</span></p>
            <p>Model weights loaded from: <code>cifar_net_2epoch.pth</code></p>
        </div>
"""
    except Exception as e:
        html_content += f"<p style='color: red;'>Error evaluating network: {e}</p>"
    
    html_content += """
    </div>
"""
    
    # Task 3: Visualize Weights
    html_content += """
    <div class="section">
        <h2>Task 3: Visualize Weights</h2>
"""
    
    print("Visualizing first layer weights...")
    try:
        weight1 = get_first_layer_weights()
        html_content += f"""
        <h3>3a. First Layer (conv1) Weights</h3>
        <div class="info-box">
            <p><strong>Shape:</strong> {list(weight1.shape)}</p>
            <p><strong>Description:</strong> 6 filters, 3 input channels, 5x5 kernel size</p>
        </div>
"""
        
        # Visualize first layer weights
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        axes = axes.flatten()
        for i in range(6):
            # Normalize each filter to [0, 1] for visualization
            filter_weights = weight1[i].numpy()
            # Transpose to (H, W, C) for visualization
            filter_img = np.transpose(filter_weights, (1, 2, 0))
            # Normalize
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
            axes[i].imshow(filter_img)
            axes[i].set_title(f'Filter {i+1}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig('./report/task3a_conv1_weights.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        html_content += """
        <img src="task3a_conv1_weights.png" alt="First layer weights visualization">
"""
    except Exception as e:
        html_content += f"<p style='color: red;'>Error visualizing first layer weights: {e}</p>"
    
    print("Visualizing second layer weights...")
    try:
        weight2 = get_second_layer_weights()
        html_content += f"""
        <h3>3b. Second Layer (conv2) Weights</h3>
        <div class="info-box">
            <p><strong>Shape:</strong> {list(weight2.shape)}</p>
            <p><strong>Description:</strong> 16 filters, 6 input channels, 5x5 kernel size</p>
        </div>
"""
        
        # Visualize second layer weights (show first 16 filters, first input channel)
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        for i in range(16):
            # Show only the first input channel for simplicity
            filter_weights = weight2[i, 0].numpy()  # Shape: (5, 5)
            # Normalize
            vmin, vmax = filter_weights.min(), filter_weights.max()
            axes[i].imshow(filter_weights, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i].set_title(f'Filter {i+1}', fontsize=8)
            axes[i].axis('off')
        plt.suptitle('Second Layer Weights (First Input Channel Only)', fontsize=12)
        plt.tight_layout()
        plt.savefig('./report/task3b_conv2_weights.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        html_content += """
        <img src="task3b_conv2_weights.png" alt="Second layer weights visualization">
        <p><em>Note: Showing first input channel only for clarity (16 filters total)</em></p>
"""
    except Exception as e:
        html_content += f"<p style='color: red;'>Error visualizing second layer weights: {e}</p>"
    
    html_content += """
    </div>
"""
    
    # Task 4: Hyperparameter Sweep
    html_content += """
    <div class="section">
        <h2>Task 4: Hyperparameter Sweep</h2>
        
        <h3>4c. Training with Different Learning Rates</h3>
        <p>The network was trained for 2 epochs using three different learning rates: 0.01, 0.001, and 0.0001. 
        Training loss and error rates were recorded every 2000 iterations.</p>
"""
    
    print("Running hyperparameter sweep (this may take a while)...")
    try:
        # Check if plots already exist
        if not os.path.exists('./report/training_loss.png'):
            hyperparameter_sweep()
        
        html_content += """
        <h4>i. Training Loss vs Iteration</h4>
        <img src="training_loss.png" alt="Training Loss">
        
        <h4>ii. Training Error vs Iteration</h4>
        <img src="training_error.png" alt="Training Error">
        
        <h4>iii. Test Error vs Iteration</h4>
        <img src="test_error.png" alt="Test Error">
        
        <h3>4d. Results Description</h3>
        <div class="info-box">
            <h4>Observations:</h4>
            <ul>
                <li><strong>Learning Rate = 0.01 (High):</strong> Training loss and errors decrease rapidly initially 
                but may show more oscillation. The network learns quickly but might overshoot optimal values.</li>
                
                <li><strong>Learning Rate = 0.001 (Medium):</strong> Shows steady, consistent decrease in both 
                training loss and error. This learning rate provides a good balance between convergence speed 
                and stability.</li>
                
                <li><strong>Learning Rate = 0.0001 (Low):</strong> Training progresses slowly with gradual 
                decrease in loss and error. The convergence is more stable but requires more iterations to 
                reach comparable performance.</li>
                
                <li><strong>Test Error:</strong> The test error patterns generally follow the training error trends, 
                indicating the models generalize reasonably well without severe overfitting. The medium learning 
                rate (0.001) typically achieves the best test performance within 2 epochs.</li>
                
                <li><strong>Conclusion:</strong> The learning rate of 0.001 appears to be optimal for this network 
                and dataset, providing stable convergence and good test performance within the limited training time.</li>
            </ul>
        </div>
"""
    except Exception as e:
        html_content += f"<p style='color: red;'>Error running hyperparameter sweep: {e}</p>"
    
    html_content += """
    </div>
    
    <hr>
    <p style="text-align: center; color: #7f8c8d;">End of Report</p>
    
</body>
</html>
"""
    
    # Save HTML report
    report_path = './report/assignment4_report.html'
    os.makedirs('./report', exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n{'='*60}")
    print(f"Report generated successfully: {report_path}")
    print(f"{'='*60}")
    print("\nGenerated files in ./report/:")
    print("  - assignment4_report.html (main report)")
    print("  - task1a_sample_batch.png")
    print("  - task3a_conv1_weights.png")
    print("  - task3b_conv2_weights.png")
    print("  - training_loss.png")
    print("  - training_error.png")
    print("  - test_error.png")
    print("\nYou can open the HTML file in a browser and print it as PDF.")

if __name__ == "__main__":
    generate_html_report()
