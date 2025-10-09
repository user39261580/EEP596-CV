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

    plt.imshow(self.cat_eye, cmap='gray')
    plt.title("Original Image")
    plt.show()

    kernel = np.array([0.25, 0.5, 0.25]) # 1D Gaussian kernel
    image = self.cat_eye.copy()
    h, w = image.shape
    self.blurred_images = []
    
    # Apply convolution 5 times
    for i in range(5):
      # Horizontal convolution
      new_image = np.zeros_like(image, dtype=np.float32)
      for y in range(h):
          for x in range(w):
              # Skip the border pixels
              if x == 0:
                new_image[y, x] = kernel[1] * image[y, x] + kernel[2] * image[y, x+1]
              elif x == w-1:
                new_image[y, x] = kernel[0] * image[y, x-1] + kernel[1] * image[y, x]
              else:
                new_image[y, x] = kernel[0] * image[y, x-1] + kernel[1] * image[y, x] + kernel[2] * image[y, x+1]
      image = np.round(new_image).astype(np.uint8)
      
      # Vertical convolution
      new_image = np.zeros_like(image, dtype=np.float32)
      for y in range(h):
          for x in range(w):
              # Skip the border pixels
              if y == 0:
                new_image[y, x] = kernel[1] * image[y, x] + kernel[2] * image[y+1, x]
              elif y == h-1:
                new_image[y, x] = kernel[0] * image[y-1, x] + kernel[1] * image[y, x]
              else:
                new_image[y, x] = kernel[0] * image[y-1, x] + kernel[1] * image[y, x] + kernel[2] * image[y+1, x]
      image = np.round(new_image).astype(np.uint8)
      
      # Store the blurred image
      self.blurred_images.append(image)

    plt.imshow(image, cmap='gray')
    plt.title("Blurred Image")
    plt.show()
      
    #cv2.imwrite(f'gaussain blur {i}.jpg', image)
    return self.blurred_images

if __name__ == "__main__":
    ass = ComputerVisionAssignment()
    # # Task 1 floodfill
    floodfill_img = ass.floodfill((100, 100))

    # Task 2 Convolution for Gaussian smoothing.
    blurred_imgs = ass.gaussian_blur()

    # # Task 3 Convolution for differentiation along the vertical direction
    # vertical_derivative = ass.gaussian_derivative_vertical()

    # # Task 4 Differentiation along another direction along the horizontal direction
    # horizontal_derivative = ass.gaussian_derivative_horizontal()

    # # Task 5 Gradient magnitude.
    # Gradient_magnitude = ass.gradient_magnitute()

    # # Task 6 Built-in convolution
    # scipy_convolve = ass.scipy_convolve()

    # # Task 7 Repeated box filtering
    # box_filter = ass.box_filter(5)
