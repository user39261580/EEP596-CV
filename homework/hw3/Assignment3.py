import numpy as np
import torch
import torchvision
import cv2 as cv
import matplotlib.pyplot as plt


class Assignment3:
    def __init__(self) -> None:
        pass

    def torch_image_conversion(self, torch_img):
        # Convert BGR to RGB
        img_rgb = cv.cvtColor(torch_img, cv.COLOR_BGR2RGB)
        torch_img = torch.from_numpy(img_rgb).float()
        
        # # Visualize the result
        # print(f"Torch tensor shape: {torch_img.shape}")
        # print(f"Torch tensor dtype: {torch_img.dtype}")
        # print(f"Min value: {torch_img.min().item()}, Max value: {torch_img.max().item()}")
        # plt.figure(figsize=(10, 8))
        # plt.imshow(torch_img.numpy().astype(np.uint8))
        # plt.title("Converted PyTorch Image (RGB)")
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

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

        return image_norm

    def Imagenet_norm(self, img):

        return ImageNet_norm

    def dimension_rearrange(self, img):

        return rearrange

    # def chain_rule(self, x, y, z):

    #     return df_dx, df_dy, df_dz, df_dq

    # def relu(self, x, w):

    #     return dx, dw


if __name__ == "__main__":
    img = cv.imread("original_image.png")
    assign = Assignment3()
    torch_img = assign.torch_image_conversion(img)
    bright_img = assign.brighten(torch_img)
    saturated_img = assign.saturation_arithmetic(img)
    noisy_img = assign.add_noise(torch_img)
    image_norm = assign.normalization_image(img)
    ImageNet_norm = assign.Imagenet_norm(img)
    rearrange = assign.dimension_rearrange(img)
    # df_dx, df_dy, df_dz, df_dq = assign.chain_rule(x=-2.0, y=5.0, z=-4.0)
    # dx, dw = assign.relu(x=[-1.0, 2.0], w=[2.0, -3.0, -3.0])
