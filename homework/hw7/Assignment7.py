import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import convolve2d  # TODO: use torch.nn.functional
import os

# region DEBUG
DEBUG = False


def print_debug(*args, **kwargs):
    if DEBUG:
        print(args, kwargs)


if __name__ == "__main__":
    DEBUG = os.environ.get("PYTHON_DEBUG_MODE")
    if DEBUG is not None and DEBUG.lower() == "true":
        DEBUG = True
        print("DEBUG mode is enabled")
# endregion


def load_image_in_grayscale(filepath) -> torch.tensor:
    return cv.imread(filepath, cv.IMREAD_GRAYSCALE)


def sum_of_abs_diff(nparray1: np.array, nparray2: np.array) -> int:
    return (np.abs(nparray1 - nparray2)).sum().item()


def scanlines(tb_left: np.array, tb_right: np.array):
    row_idx = 152
    col_idx1 = 102
    col_len = 100
    tb_left_cropped = tb_left[row_idx][col_idx1 : col_idx1 + col_len]

    g_best = None
    d_best = None
    for d in range(col_len + 1):  # TODO: check max disparity
        tb_right_cropped = tb_right[row_idx][col_idx1 - d : col_idx1 - d + col_len]
        g = sum_of_abs_diff(tb_left_cropped, tb_right_cropped)
        if g_best == None or g < g_best:
            g_best, d_best = g, d

    return d_best


def plot_1d_array(array, title, xlabel=None, ylabel=None, save_image=True):
    domain = range(len(array))
    plt.plot(domain, array, marker="o")
    plt.xlabel(title)
    plt.ylabel(xlabel)
    plt.title(ylabel)
    plt.grid(True)
    if save_image:
        plt.savefig(f"figure/{title}.png")
    plt.show()


def plot_2d_array_as_image(array2d: np.array, title, save_image=True):
    plt.imshow(array2d, cmap="gray")
    plt.title(title)
    plt.colorbar()
    if save_image:
        plt.savefig(f"figure/{title}.png")
    plt.show()


def shift_array(nparray: np.array, d: int) -> np.array:
    shifted = np.zeros_like(nparray)
    if d == 0:
        shifted[:, :] = nparray[:, :]
    elif d > 0:
        shifted[:, d:] = nparray[:, :-d]
    elif d < 0:
        shifted[:, : nparray.shape[1] + d] = nparray[:, -d:]
    return shifted


if DEBUG:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert (shift_array(a, 1) == [[0, 1, 2], [0, 4, 5], [0, 7, 8]]).all()
    assert (shift_array(a, 2) == [[0, 0, 1], [0, 0, 4], [0, 0, 7]]).all()


def auto_correlation(tb_right):
    max_d = 30
    auto_correlations = []
    for d in range(max_d + 1):
        abs_diff_image = np.abs(tb_right - shift_array(tb_right, d))
        auto_correlations.append(abs_diff_image[152][152])

    if DEBUG:
        plot_1d_array(auto_correlations, auto_correlation.__name__)
    return auto_correlations


# TODO
def convolve2d_torch(array: np.array, kernel_size: int):
    as_tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor(np.ones((kernel_size, kernel_size)), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    convolved = nn.functional.conv2d(as_tensor, kernel, padding=kernel_size // 2)
    if DEBUG:
        assert convolved.shape == as_tensor.shape

    return np.array(convolved.squeeze().squeeze())


def smoothing(tb_right):
    max_d = 30
    smoothed_auto_correlations = []
    kernel_size = 5

    for d in range(max_d + 1):
        # Create shifted right image
        shifted_right = shift_array(tb_right, d)
        
        # Calculate the absolute difference between the original right image and the shifted right image
        abs_diff_image = np.abs(tb_right - shifted_right)
        
        # Use 5x5 box filter for smoothing (convolution)
        smoothed_diff_image = convolve2d_torch(abs_diff_image, kernel_size)
        
        # Extract the smoothed auto-correlation value at pixel (152, 152)
        smoothed_auto_correlations.append(smoothed_diff_image[152][152])

    if DEBUG:
        plot_1d_array(smoothed_auto_correlations, "Smoothed Auto-correlation", save_image=False)
        plot_2d_array_as_image(smoothed_diff_image, "Smoothed Auto-correlation Image", save_image=False)
        
    return smoothed_auto_correlations

def cross_correlation(tb_left, tb_right):
    max_d = 30
    kernel_size = 5
    height, width = tb_left.shape
    
    # Create a 3D array to store all smoothed cross-correlation images
    smoothed_cross_correlation_tensor = np.zeros((height, width, max_d + 1))
    cross_correlation_values_at_152 = []

    for d in range(max_d + 1):
        # Create shifted right image
        shifted_right = shift_array(tb_right, d)
        
        # Calculate the absolute difference between the original left image and the shifted right image
        abs_diff_image = np.abs(tb_left - shifted_right)
        
        # Use 5x5 box filter for smoothing
        smoothed_diff = convolve2d_torch(abs_diff_image, kernel_size)
        
        # Store the smoothed result in the 3D array
        smoothed_cross_correlation_tensor[:, :, d] = smoothed_diff
        
        # Extract the value at (152, 152) for plotting
        cross_correlation_values_at_152.append(smoothed_diff[152][152])

    if DEBUG:
        plot_1d_array(cross_correlation_values_at_152, "Cross-correlation at (152, 152)", save_image=False)
        plot_2d_array_as_image(smoothed_diff, "Smoothed Cross-correlation Image", save_image=False)
        
    return cross_correlation_values_at_152

def disparity_map(tb_left, tb_right, plot_result=False):
    max_d = 30
    kernel_size = 5
    height, width = tb_left.shape
    
    # Create a 3D array to store all smoothed cross-correlation images
    smoothed_cross_correlation_tensor = np.zeros((height, width, max_d + 1))

    for d in range(max_d + 1):
        # Create shifted right image
        shifted_right = shift_array(tb_right, d)
        
        # Calculate the absolute difference between the original left image and the shifted right image
        abs_diff_image = np.abs(tb_left - shifted_right)
        
        # Use 5x5 box filter for smoothing
        smoothed_diff = convolve2d_torch(abs_diff_image, kernel_size)
        
        # Store the smoothed result in the 3D array
        smoothed_cross_correlation_tensor[:, :, d] = smoothed_diff
    
    # For each pixel, find the disparity d that minimizes the smoothed cross-correlation value
    # np.argmin returns the index of the minimum value, which corresponds to the disparity d
    left_right_disparity = np.argmin(smoothed_cross_correlation_tensor, axis=2)
    
    if plot_result:
        plot_2d_array_as_image(left_right_disparity, "Left-Right Disparity Map", save_image=False)
        
    return left_right_disparity.astype(np.uint8)


def right_left_disparity(tb_left, tb_right, plot_result=False):
    max_d = 30
    height, width = tb_left.shape
    
    cost_tensor = np.zeros((max_d + 1, height, width), dtype=np.float32)

    for d in range(max_d + 1):
        # Shift the left image
        shifted_left = shift_array(tb_left, d)
        
        # Calculate the absolute difference between the original right image and the shifted left image
        abs_diff_image = np.abs(tb_right - shifted_left)
        cost_tensor[d] = abs_diff_image

    right_left_disparity_map = np.argmin(cost_tensor, axis=0)
    
    if plot_result:
        plot_2d_array_as_image(right_left_disparity_map, "Right-Left Disparity Map", save_image=False)
        
    return right_left_disparity_map.astype(np.uint8)



def disparity_check(tb_left, tb_right, plot_result=False):
    disp_L = disparity_map(tb_left, tb_right, plot_result=False)
    disp_R = right_left_disparity(tb_left, tb_right, plot_result=False)

    H, W = disp_L.shape
    
    cleaned = np.zeros_like(disp_L, dtype=np.float32)
    for y in range(H):
        for x in range(W):
            d = int(disp_L[y, x])
            xr = x - d
            if d >= 0 and xr >= 0 and xr < W:
                # Check if the disparity matches
                if int(disp_R[y, xr]) == d:
                    cleaned[y, x] = d
            # else leave as 0 (invalid disparity)
                    
    if plot_result:
        show = (cleaned * (255.0 / 30)).astype(np.uint8)
        plot_2d_array_as_image(show, "disparity_cleaned_visual", save_image=False)
    return cleaned



def reconstruction(tb_left, tb_right):
    # Fill your code hear
    pass


if __name__ == "__main__":
    DEBUG = True
    tb_left = load_image_in_grayscale("tsukuba_left.png")
    tb_right = load_image_in_grayscale("tsukuba_right.png")
    # scanlines(tb_left, tb_right)
    # auto_correlation(tb_right)
    # smoothing(tb_right)
    # cross_correlation(tb_left, tb_right)
    # disparity_map(tb_left, tb_right, plot_result=True)
    # right_left_disparity(tb_left, tb_right, plot_result=True)
    disparity_check(tb_left, tb_right, plot_result=True)