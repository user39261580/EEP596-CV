import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from datetime import datetime


class Assignment3:
    def __init__(self) -> None:
        pass

    def torch_image_conversion(self, torch_img):
        img_rgb = cv.cvtColor(torch_img, cv.COLOR_BGR2RGB) # Convert BGR to RGB
        torch_img = torch.from_numpy(img_rgb).float() # Convert to float tensor
        
        return torch_img

    def brighten(self, torch_img):
        # Add a constant brightness value and clamp to valid image range
        bright_img = torch_img + 100

        return bright_img

    def saturation_arithmetic(self, img):
        torch_img = self.torch_image_conversion(img).to(torch.uint8) # Convert to uint8
        bright_img = self.brighten(torch_img)
        saturated_img = torch.clamp(bright_img, 0, 255) # Clamp to [0, 255]

        return saturated_img

    def add_noise(self, torch_img):
        noise = torch.randn(torch_img.shape) * 100.0 # Generate Gaussian noise
        noisy_img = torch_img + noise
        noisy_img = noisy_img / 255.0 # Range [0,1] expected
        noisy_img = torch.clamp(noisy_img, 0, 1) # Clamp to [0, 1]
        
        return noisy_img

    def normalization_image(self, img):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).double() / 255.0 # Convert to float64 tensor and normalize to [0, 1]

        # Calculate mean and std for each channel
        mean = torch.mean(img_tensor, dim=(0, 1))
        std = torch.std(img_tensor, dim=(0, 1))

        image_norm = (img_tensor - mean) / std # Standardization
        image_norm = torch.clamp(image_norm, -1, 1) # Limit range

        return image_norm

    def Imagenet_norm(self, img):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).double() / 255.0 # Convert to float64 tensor and normalize to [0, 1]

        # Mean and std from ImageNet dataset
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float64)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float64)

        ImageNet_norm = (img_tensor - mean) / std # Standardization
        ImageNet_norm = torch.clamp(ImageNet_norm, -1, 1)  # Limit range

        return ImageNet_norm

    def dimension_rearrange(self, img):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float() # Convert to float tensor
        
        rearrange = img_tensor.permute(2, 0, 1) # HxWxC -> CxHxW
        rearrange = rearrange.unsqueeze(0) # CxHxW -> NxCxHxW (N=1)
        
        return rearrange
    
    def stride(self, img):
        # Define Scharr_x filter and flip it for convolution
        scharr_x = torch.tensor([[-3, 0, 3],
                                [-10, 0, 10],
                                [-3, 0, 3]], dtype=torch.float32)
        scharr_x = torch.flip(scharr_x, dims=[0, 1])
        
        img_tensor = torch.from_numpy(img).float() # Convert to float tensor
        
        # Reshape to (1, 1, H, W) for conv2d
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        kernel = scharr_x.unsqueeze(0).unsqueeze(0)
        
        strided_img = torch.nn.functional.conv2d(img_tensor, kernel, stride=2, padding=1) # Apply conv as: stride=2, padding=1
        
        strided_img = strided_img.squeeze(0).squeeze(0) # Remove batch and channel dimensions

        return strided_img


def tensor_to_base64(tensor_img, normalize=False):
    """Convert a tensor to base64 encoded image"""
    # Convert tensor to numpy
    if isinstance(tensor_img, torch.Tensor):
        img_np = tensor_img.cpu().numpy()
    else:
        img_np = tensor_img
    
    # Handle different data types
    if img_np.dtype == np.float64 or img_np.dtype == np.float32:
        if normalize:
            # For normalized images, scale to 0-255
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        else:
            # For ImageNet norm, scale from [-1, 1] to 0-255
            img_np = np.clip((img_np + 1) / 2 * 255, 0, 255).astype(np.uint8)
    else:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    # Handle different channel configurations
    if len(img_np.shape) == 3:
        if img_np.shape[0] == 3:  # CxHxW format
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.shape[2] == 3:  # RGB format
            img_np = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)
    elif len(img_np.shape) == 4:  # NxCxHxW format
        # Extract first image and transpose
        img_np = img_np[0]  # Remove batch dimension
        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)
    
    # Encode to PNG
    success, buffer = cv.imencode('.png', img_np)
    if not success:
        raise RuntimeError("Failed to encode image to PNG")
    img_base64 = base64.b64encode(buffer).decode()
    return img_base64


def generate_html_report(saturated_img, noisy_img, imagenet_norm, rearrange, strided_img):
    """Generate HTML report with all task outputs"""
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Assignment 3 Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 0.9em;
        }}
        .section {{
            margin-bottom: 40px;
            page-break-inside: avoid;
        }}
        .section-title {{
            font-size: 1.8em;
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .task {{
            margin-bottom: 30px;
            padding: 20px;
            background-color: #ecf0f1;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .task-name {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        .image-container {{
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .info {{
            background-color: #d5dbdb;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            font-family: monospace;
            font-size: 0.9em;
            overflow-x: auto;
        }}
        .info-title {{
            font-weight: bold;
            color: #1a5276;
            margin-bottom: 8px;
        }}
        @media print {{
            body {{
                background-color: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
                padding: 0;
            }}
            .section {{
                page-break-inside: avoid;
            }}
            .task {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Assignment 3 Report</h1>
        <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <div class="section">
            <div class="section-title">Task Outputs</div>
            
            <div class="task">
                <div class="task-name">1. Saturated Image (saturated_img)</div>
                <div class="image-container">
                    <img src="data:image/png;base64,{tensor_to_base64(saturated_img)}" alt="Saturated Image">
                </div>
                <div class="info">
                    <div class="info-title">Tensor Information:</div>
                    Shape: {saturated_img.shape}<br>
                    Dtype: {saturated_img.dtype}<br>
                    Min Value: {saturated_img.min().item():.4f}<br>
                    Max Value: {saturated_img.max().item():.4f}<br>
                    Mean Value: {saturated_img.float().mean().item():.4f}
                </div>
            </div>
            
            <div class="task">
                <div class="task-name">2. Noisy Image (noisy_img)</div>
                <div class="image-container">
                    <img src="data:image/png;base64,{tensor_to_base64(noisy_img, normalize=True)}" alt="Noisy Image">
                </div>
                <div class="info">
                    <div class="info-title">Tensor Information:</div>
                    Shape: {noisy_img.shape}<br>
                    Dtype: {noisy_img.dtype}<br>
                    Min Value: {noisy_img.min().item():.4f}<br>
                    Max Value: {noisy_img.max().item():.4f}<br>
                    Mean Value: {noisy_img.mean().item():.4f}
                </div>
            </div>
            
            <div class="task">
                <div class="task-name">3. ImageNet Normalized Image (ImageNet_norm)</div>
                <div class="image-container">
                    <img src="data:image/png;base64,{tensor_to_base64(imagenet_norm)}" alt="ImageNet Normalized">
                </div>
                <div class="info">
                    <div class="info-title">Tensor Information:</div>
                    Shape: {imagenet_norm.shape}<br>
                    Dtype: {imagenet_norm.dtype}<br>
                    Min Value: {imagenet_norm.min().item():.4f}<br>
                    Max Value: {imagenet_norm.max().item():.4f}<br>
                    Mean Value: {imagenet_norm.mean().item():.4f}
                </div>
            </div>
            
            <div class="task">
                <div class="task-name">4. Rearranged Dimensions (rearrange)</div>
                <div class="image-container">
                    <img src="data:image/png;base64,{tensor_to_base64(rearrange)}" alt="Rearranged">
                </div>
                <div class="info">
                    <div class="info-title">Tensor Information:</div>
                    Shape (NxCxHxW): {rearrange.shape}<br>
                    Original Shape (HxWxC) before rearrangement: (H, W, 3)<br>
                    Dtype: {rearrange.dtype}<br>
                    Min Value: {rearrange.min().item():.4f}<br>
                    Max Value: {rearrange.max().item():.4f}<br>
                    Mean Value: {rearrange.mean().item():.4f}
                </div>
            </div>
            
            <div class="task">
                <div class="task-name">5. Strided Image (strided_img)</div>
                <div class="image-container">
                    <img src="data:image/png;base64,{tensor_to_base64(strided_img)}" alt="Strided Image">
                </div>
                <div class="info">
                    <div class="info-title">Tensor Information:</div>
                    Shape: {strided_img.shape}<br>
                    Dtype: {strided_img.dtype}<br>
                    Min Value: {strided_img.min().item():.4f}<br>
                    Max Value: {strided_img.max().item():.4f}<br>
                    Mean Value: {strided_img.mean().item():.4f}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    return html_content


if __name__ == "__main__":
    # Create report directory if it doesn't exist
    report_dir = "report"
    os.makedirs(report_dir, exist_ok=True)
    
    # Load images
    img = cv.imread("original_image.PNG")
    cat_eye = cv.imread('cat_eye.jpg', cv.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Could not load original_image.PNG")
        exit(1)
    if cat_eye is None:
        print("Error: Could not load cat_eye.jpg")
        exit(1)
    
    # Create assignment instance
    assign = Assignment3()
    
    # Generate all outputs
    torch_img = assign.torch_image_conversion(img)
    saturated_img = assign.saturation_arithmetic(img)
    noisy_img = assign.add_noise(torch_img)
    imagenet_norm = assign.Imagenet_norm(img)
    rearrange = assign.dimension_rearrange(img)
    strided_img = assign.stride(cat_eye)
    
    # Generate HTML report
    html_content = generate_html_report(saturated_img, noisy_img, imagenet_norm, rearrange, strided_img)
    
    # Save report
    report_path = os.path.join(report_dir, "assignment3_report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Report generated successfully: {report_path}")
    print(f"Full path: {os.path.abspath(report_path)}")
