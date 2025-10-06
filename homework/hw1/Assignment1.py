import cv2
import numpy as np
import os


class ComputerVisionAssignment:
    def __init__(self, image_path, binary_image_path):
        self.image = cv2.imread(image_path)
        self.binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    def check_package_versions(self):
        # Ungraded
        import numpy as np
        import matplotlib
        import cv2

        # print(np.__version__)
        # print(matplotlib.__version__)
        # print(cv2.__version__)

    def load_and_analyze_image(self):
        Image_data_type = type(self.image)
        Pixel_data_type = self.image.dtype
        Image_shape = self.image.shape

        # print(f"Image data type: {Image_data_type}")
        # print(f"Pixel data type: {Pixel_data_type}")
        # print(f"Image dimensions: {Image_shape}")

        return Image_data_type, Pixel_data_type, Image_shape

    def create_red_image(self):
        red_image = self.image.copy()
        red_image[:, :, 0] = 0 
        red_image[:, :, 1] = 0

        return red_image

    def create_photographic_negative(self):
        negative_image = np.subtract(255, self.image)

        return negative_image

    def swap_color_channels(self):
        red = self.image[:, :, 2]
        blue = self.image[:, :, 0]
        swapped_image = self.image.copy()

        swapped_image[:, :, 0] = red
        swapped_image[:, :, 2] = blue

        return swapped_image

    def foliage_detection(self):
        blue = self.image[:, :, 0]
        green = self.image[:, :, 1]
        red = self.image[:, :, 2]
        
        # Create mask where green >= 50 AND red < 50 AND blue < 50
        mask = (green >= 50) & (red < 50) & (blue < 50)
        
        # Create binary image and set data type to uint8
        foliage_image = np.where(mask, 255, 0).astype(np.uint8)

        return foliage_image

    def shift_image(self):
        # Shift right
        shifted_image = np.roll(self.image, shift=200, axis=1)
        shifted_image[:, :200] = 0

        # Shift down
        shifted_image = np.roll(shifted_image, shift=100, axis=0)
        shifted_image[:100, :] = 0

        return shifted_image

    def rotate_image(self):
        # Rotate image by 90*k degrees
        rotated_image = np.rot90(self.image, k=-1)

        return rotated_image

    def similarity_transform(self, scale, theta, shift):
        # Implement with OpenCV
        height, width = self.image.shape[:2]
        center = (0, 0)
        rotation_matrix = cv2.getRotationMatrix2D(center, -theta, scale)
        rotation_matrix[0, 2] += shift[0]
        rotation_matrix[1, 2] += shift[1]
        transformed_image = cv2.warpAffine(self.image, rotation_matrix, (width, height), flags=cv2.INTER_NEAREST)

        return transformed_image

    def convert_to_grayscale(self):
        conversion_matrix = np.array([0.1, 0.6, 0.3])  # B, G, R weights
        gray_image = np.dot(self.image, conversion_matrix)
        gray_image = np.round(gray_image).astype(np.uint8)
        print(gray_image.shape)
    
        return gray_image

    def compute_moments(self):
        # Use the grayscale intensity values (0-255) as the image 'mass' for raw moments.
        # This makes M00 the sum of pixel intensities rather than a count of foreground pixels.
        mask = self.binary_image.astype(np.float64)

        # Image shape and coordinate grids
        height, width = mask.shape
        ys = np.arange(height) # Get all y coordinates
        xs = np.arange(width) # Get all x coordinates
        x_grid, y_grid = np.meshgrid(xs, ys) # Make coordinate grids

        # Raw moments
        m00 = mask.sum()
        m10 = (x_grid * mask).sum()
        m01 = (y_grid * mask).sum()
        x_bar = m10 / m00
        y_bar = m01 / m00

        # Centralized second-order moments
        dx = x_grid - x_bar
        dy = y_grid - y_bar
        mu20 = (dx ** 2 * mask).sum()
        mu02 = (dy ** 2 * mask).sum()
        mu11 = (dx * dy * mask).sum()

        # Print the results
        # print("First-Order Moments:")
        # print(f"Standard (Raw) Moments: M00 = {m00}, M10 = {m10}, M01 = {m01}")
        # print("Centralized Moments:")
        # print(f"x_bar = {x_bar}, y_bar = {y_bar}")
        # print("Second-Order Centralized Moments:")
        # print(f"mu20 = {mu20}, mu02 = {mu02}, mu11 = {mu11}")

        return m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11

    def compute_orientation_and_eccentricity(self):
        # Calculate moments
        m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11 = self.compute_moments()
        
        # Compute orientation (in radians first)
        # theta = 0.5 * arctan(2 * mu11 / (mu20 - mu02))
        theta_rad = 0.5 * np.arctan2((2 * mu11) / m00, (mu20 - mu02) / m00)
        
        # Convert to degrees and make it clockwise w.r.t. positive horizontal axis
        orientation = np.degrees(theta_rad)
        
        # Compute eccentricity
        # First compute the eigenvalues of the covariance matrix
        term1 = (mu20 + mu02) / m00 / 2
        term2 = np.sqrt(4 * (mu11 / m00)**2 + ((mu20 - mu02) / m00)**2) / 2
        lambda1 = term1 + term2  # Major axis
        lambda2 = term1 - term2  # Minor axis
        
        # Eccentricity: e = sqrt(1 - (b/a)^2) where a >= b
        if lambda1 > 0:
            eccentricity = np.sqrt(1 - lambda2 / lambda1)
        else:
            eccentricity = 0
        
        # Load the image as BGR (use self.image which is already loaded in __init__)
        # self.image should be loaded as BGR when the object is created
        assert self.image is not None, "Image not loaded properly"
        glasses_bgr = self.image.copy()
        
        # Prepare ellipse parameters
        center = (int(x_bar), int(y_bar))
        
        # Compute axes lengths from eigenvalues
        # Semi-axes are sqrt(lambda) scaled by a factor
        # Use 2*sqrt(lambda) to get axes that match the second moment
        axes = (int(2 * np.sqrt(lambda1)), int(2 * np.sqrt(lambda2)))
        
        # Draw the ellipse in red (BGR: 0, 0, 255) with thickness 1
        glasses_with_ellipse = glasses_bgr.copy()
        cv2.ellipse(glasses_with_ellipse, center, axes, orientation, 0, 360, (0, 0, 255), 1)

        return orientation, eccentricity, glasses_with_ellipse


if __name__ == "__main__":

    assignment = ComputerVisionAssignment("picket_fence.png", "binary_image.png")

    # Task 0: Check package versions
    assignment.check_package_versions()

    # Task 1: Load and analyze the image
    assignment.load_and_analyze_image()

    # Task 2: Create a red image
    red_image = assignment.create_red_image()

    # Task 3: Create a photographic negative
    negative_image = assignment.create_photographic_negative()

    # Task 4: Swap color channels
    swapped_image = assignment.swap_color_channels()

    # Task 5: Foliage detection
    foliage_image = assignment.foliage_detection()

    # Task 6: Shift the image
    shifted_image = assignment.shift_image()

    # Task 7: Rotate the image
    rotated_image = assignment.rotate_image()

    # Task 8: Similarity transform
    transformed_image = assignment.similarity_transform(
        scale=2.0, theta=45.0, shift=[100, 100]
    )

    # Task 9: Grayscale conversion
    gray_image = assignment.convert_to_grayscale()

    glasses_assignment = ComputerVisionAssignment(
        "glasses_outline.png", "glasses_outline.png"
    )

    # Task 10: Moments of a binary image
    glasses_assignment.compute_moments()

    # Task 11: Orientation and eccentricity of a binary image
    orientation, eccentricity, glasses_with_ellipse = (
        glasses_assignment.compute_orientation_and_eccentricity()
    )
