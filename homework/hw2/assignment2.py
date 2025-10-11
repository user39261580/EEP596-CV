# -*- coding: utf-8 -*-
"""EEP 596 HW2
"""

import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt

class ComputerVisionAssignment():
  def __init__(self) -> None:
    self.ant_img = cv2.imread('ant_outline.png')
    self.cat_eye = cv2.imread('cat_eye.jpg', cv2.IMREAD_GRAYSCALE)

  def __convolve_1d(self, image, kernel, axis):
    """
    Convolution with a [3x1] kernel along specified axis (0 for vertical, 1 for horizontal).
    """
    h, w = image.shape
    new_image = np.zeros_like(image, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            if axis == 1:  # horizontal
                if x == 0:
                    new_image[y, x] = kernel[1] * image[y, x] + kernel[2] * image[y, x+1]
                elif x == w-1:
                    new_image[y, x] = kernel[0] * image[y, x-1] + kernel[1] * image[y, x]
                else:
                    new_image[y, x] = kernel[0] * image[y, x-1] + kernel[1] * image[y, x] + kernel[2] * image[y, x+1]
            elif axis == 0:  # vertical
                if y == 0:
                    new_image[y, x] = kernel[1] * image[y, x] + kernel[2] * image[y+1, x]
                elif y == h-1:
                    new_image[y, x] = kernel[0] * image[y-1, x] + kernel[1] * image[y, x]
                else:
                    new_image[y, x] = kernel[0] * image[y-1, x] + kernel[1] * image[y, x] + kernel[2] * image[y+1, x]
    return new_image

  def __convolve_separable(self, image, kernel1, kernel2, order='hv'):
    if order == 'hv':
        temp = self.__convolve_1d(image, kernel1, axis=1)
        result = self.__convolve_1d(temp, kernel2, axis=0)
    elif order == 'vh':
        temp = self.__convolve_1d(image, kernel1, axis=0)
        result = self.__convolve_1d(temp, kernel2, axis=1)
    return result

  def floodfill(self, seed = (0, 0)):

    # Define the fill color (e.g., bright green)
    fill_color = (0, 0, 255)  # (B, G, R)

    # Create a copy of the input image to keep the original image unchanged
    # output_image = cv2.cvtColor(self.ant_img.copy(), cv2.COLOR_BGR2GRAY)
    output_image = self.ant_img.copy()

    # Define a stack for floodfill
    h, w = output_image.shape[:2]
    if seed[0] < 0 or seed[0] >= h or seed[1] < 0 or seed[1] >= w:
        return output_image
    original_color = output_image.copy()[seed[0], seed[1]]

    if np.array_equal(original_color, fill_color):
        return output_image
    
    # BFS floodfill
    stack = [seed]
    while stack:
        x, y = stack.pop()
        if x < 0 or x >= h or y < 0 or y >= w:
            continue
        if not np.array_equal(output_image[x, y], original_color):
            continue
        output_image[x, y] = fill_color
        stack.append((x+1, y))
        stack.append((x-1, y))
        stack.append((x, y+1))
        stack.append((x, y-1))

    #cv2.imwrite('floodfille.jpg', output_image)
    return output_image

  def gaussian_blur(self):
    """
    Apply Gaussian blur to the image iteratively.
    """

    kernel = np.array([0.25, 0.5, 0.25]) # 1D Gaussian kernel
    image = self.cat_eye.copy()
    h, w = image.shape
    self.blurred_images = []
    
    # Apply convolution 5 times
    for i in range(5):
        image = self.__convolve_separable(image, kernel, kernel, 'hv')
        image = np.round(image).astype(np.uint8)
        self.blurred_images.append(image)
      
    # plt.imshow(self.blurred_images[4], cmap='gray')
    # plt.title('Gaussain blur after 5 iterations')
    # plt.show()
    #cv2.imwrite(f'gaussain blur {i}.jpg', image)
    return self.blurred_images

  def gaussian_derivative_vertical(self):
    # Define kernels
    kernel_h = np.array([0.25, 0.5, 0.25])  # Horizontal smoothing: 0.25 * [1 2 1]
    kernel_v = np.array([0.5, 0, -0.5])     # Vertical derivative: 0.5 * [1 0 -1], flipped for convolution
    
    # Store images
    self.vDerive_images = []
    for i in range(5):
      # Get the blurred image from previous step
      blurred_image = self.blurred_images[i]
      h, w = blurred_image.shape
      
      result_image = self.__convolve_separable(blurred_image, kernel_h, kernel_v, 'hv')
      
      # Convert to uint8: scale by 2, add offset 127, and clamp to [0, 255]
      result_image = 2 * result_image + 127
      result_image = np.clip(result_image, 0, 255)
      image = result_image.astype(np.uint8)
      
      self.vDerive_images.append(image)
      #cv2.imwrite(f'vertical {i}.jpg', image)

    # plt.imshow(self.vDerive_images[4], cmap='gray')
    # plt.title('Vertical Derivative after 5 iterations')
    # plt.show()
    return self.vDerive_images

  def gaussian_derivative_horizontal(self):
    #Define kernels
    kernel_h = np.array([0.5, 0, -0.5])     # Horizontal derivative: 0.5 * [1 0 -1], flipped
    kernel_v = np.array([0.25, 0.5, 0.25])  # Vertical smoothing: 0.25 * [1 2 1] for convolution

    # Store images after computing horizontal derivative
    self.hDerive_images = []

    for i in range(5):
      # Get the blurred image from previous step
      blurred_image = self.blurred_images[i]
      h, w = blurred_image.shape
      
      result_image = self.__convolve_separable(blurred_image, kernel_v, kernel_h, 'vh')
      
      
      # Convert to uint8: scale by 2, add offset 127, and clamp to [0, 255]
      result_image = 2 * result_image + 127
      result_image = np.clip(result_image, 0, 255)
      image = result_image.astype(np.uint8)

      self.hDerive_images.append(image)
      #cv2.imwrite(f'horizontal {i}.jpg', image)

    # plt.imshow(self.hDerive_images[4], cmap='gray')
    # plt.title('Horizontal Derivative after 5 iterations')
    # plt.show()
    return self.hDerive_images

  def gradient_magnitute(self):
    # Store the computed gradient magnitute
    self.gdMagnitute_images =[]

    # Define kernels for vDerive
    vDerive_kernel_h = np.array([0.25, 0.5, 0.25])  # Horizontal smoothing: 0.25 * [1 2 1]
    vDerive_kernel_v = np.array([0.5, 0, -0.5])     # Vertical derivative: 0.5 * [1 0 -1], flipped for convolution
    
    # Store images
    vDerive_images = []
    for i in range(5):
      # Get the blurred image from previous step
      blurred_image = self.blurred_images[i]
      h, w = blurred_image.shape
      
      result_image = self.__convolve_separable(blurred_image, vDerive_kernel_h, vDerive_kernel_v, 'hv')
      
      vDerive_images.append(result_image)
      #cv2.imwrite(f'vertical {i}.jpg', image)


    #Define kernels for hDerive
    hDerive_kernel_h = np.array([0.5, 0, -0.5])     # Horizontal derivative: 0.5 * [1 0 -1], flipped
    hDerive_kernel_v = np.array([0.25, 0.5, 0.25])  # Vertical smoothing: 0.25 * [1 2 1] for convolution

    # Store images after computing horizontal derivative
    hDerive_images = []

    for i in range(5):
      # Get the blurred image from previous step
      blurred_image = self.blurred_images[i]
      h, w = blurred_image.shape
      
      result_image = self.__convolve_separable(blurred_image, hDerive_kernel_v, hDerive_kernel_h, 'vh')

      hDerive_images.append(result_image)

    for i, (vimg, himg) in enumerate(zip(vDerive_images, hDerive_images)):
      # Compute gradient magnitude using Manhattan distance: abs(gx) + abs(gy)
      gradient_magnitude = np.abs(himg) + np.abs(vimg)
      
      # Scale by 4 and clamp to [0, 255]
      gradient_magnitude = 4 * gradient_magnitude
      gradient_magnitude = np.clip(gradient_magnitude, 0, 255)
      
      # Convert to uint8
      image = gradient_magnitude.astype(np.uint8)
      
      self.gdMagnitute_images.append(image)
      #cv2.imwrite(f'gradient {i}.jpg', image)
    
    # plt.imshow(self.gdMagnitute_images[4], cmap='gray')
    # plt.title('Gradient Magnitude after 5 iterations')
    # plt.show()
    return self.gdMagnitute_images

  def scipy_convolve(self):
    # Define the 2D smoothing kernel (horizontal smoothing)
    kernel_h = np.array([[0.25, 0.5, 0.25]])  # Horizontal: 0.25 * [1 2 1]
    
    # Define the 2D differentiating kernel (vertical derivative)
    kernel_v = np.array([[0.5], [0], [-0.5]])  # Vertical: 0.5 * [1 0 -1], flipped for convolution
    
    # Combine into a 2D kernel: outer product of horizontal smoothing and vertical derivative
    kernel_2d = kernel_v @ kernel_h  # Results in a 3x3 kernel
    
    # Store outputs
    self.scipy_smooth = []

    for i in range(5):
      # Get the blurred image from previous step (same as gaussian_derivative_vertical)
      blurred_image = self.blurred_images[i]
      
      # Perform convolution using scipy with zero padding (mode='same' maintains shape)
      result_image = scipy.signal.convolve2d(blurred_image, kernel_2d, mode='same', boundary='fill', fillvalue=0)
      
      # Convert to uint8: scale by 2, add offset 127, and clamp to [0, 255]
      result_image = 2 * result_image + 127
      result_image = np.clip(result_image, 0, 255)
      image = result_image.astype(np.uint8)
      
      self.scipy_smooth.append(image)
      #cv2.imwrite(f'scipy smooth {i}.jpg', image)

    # plt.imshow(self.scipy_smooth[4], cmap='gray')
    # plt.title('Scipy Convolution after 5 iterations')
    # plt.show()
    return self.scipy_smooth

  def box_filter(self, num_repetitions):
    # Define box filter
    box_filter = [1, 1, 1]
    out = [1, 1, 1]

    for _ in range(num_repetitions):
      # Perform 1D convolution manually
      # Result will have length N + M - 1
      N = len(out)
      M = len(box_filter)
      result_length = N + M - 1
      result = [0] * result_length
      
      # Manual convolution: for each position in output
      for i in range(result_length):
        # Sum over all valid overlapping positions
        for j in range(M):
          # Check if the index is valid in the 'out' array
          out_index = i - j
          if 0 <= out_index < N:
            result[i] += out[out_index] * box_filter[j]
      
      out = result

    # plt.figure(figsize=(10, 6))
    # plt.plot(out, 'bo-', linewidth=2, markersize=8)
    # plt.title(f'Box Filter after {num_repetitions} Convolutions (Approximates Gaussian)')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.grid(True, alpha=0.3)
    # plt.show()
    return out

if __name__ == "__main__":
    ass = ComputerVisionAssignment()
    # Task 1 floodfill
    floodfill_img = ass.floodfill((100, 100))

    # Task 2 Convolution for Gaussian smoothing.
    blurred_imgs = ass.gaussian_blur()

    # Task 3 Convolution for differentiation along the vertical direction
    vertical_derivative = ass.gaussian_derivative_vertical()

    # Task 4 Differentiation along another direction along the horizontal direction
    horizontal_derivative = ass.gaussian_derivative_horizontal()

    # Task 5 Gradient magnitude.
    Gradient_magnitude = ass.gradient_magnitute()

    # Task 6 Built-in convolution
    scipy_convolve = ass.scipy_convolve()

    # # Task 7 Repeated box filtering
    box_filter = ass.box_filter(5)
