import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2
import os
import requests
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util import random_noise
import requests
from io import BytesIO
from PIL import Image
from functools import lru_cache

class WaveletDenoiser:
    def __init__(self, wavelet='bior4.4', level=None):
        """
        Initialize the wavelet denoiser
        
        Parameters:
        wavelet: Wavelet to use (default: 'bior4.4')
        level: Decomposition level (default: None, will be auto-selected)
        """
        self.wavelet = wavelet
        self.level = level
    
    def add_gaussian_noise(self, image, var=0.01):
        """Add Gaussian noise to an image"""
        return random_noise(image, mode='gaussian', var=var)
    
    def _pad_to_dyadic(self, img):
        """Pad image to have dimensions divisible by 2^level"""
        if self.level is None:
            raise ValueError("self.level must be set to an integer before padding. (Got None)")
        pad_x = int(np.ceil(img.shape[0] / (2**self.level))) * (2**self.level) - img.shape[0]
        pad_y = int(np.ceil(img.shape[1] / (2**self.level))) * (2**self.level) - img.shape[1]
        
        # Handle different numbers of channels
        if img.ndim == 2:
            return np.pad(img, ((0, pad_x), (0, pad_y)), mode='reflect')
        else:
            return np.pad(img, ((0, pad_x), (0, pad_y), (0, 0)), mode='reflect')
    
    def _get_optimal_level(self, img_shape):
        """Determine optimal decomposition level based on image size"""
        min_dim = min(img_shape[:2])
        # Rule of thumb: level shouldn't make subbands smaller than 16 pixels
        max_level = int(np.log2(min_dim / 16))
        return max(1, min(4, max_level))  # Keep between 1 and 4
    
    def _rgb_to_ycbcr(self, img_rgb):
        """Convert RGB to YCbCr color space"""
        # Ensure input is float32 and in range [0, 1]
        img_rgb = img_rgb.astype(np.float32)
        
        r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5
        return np.stack([y, cb, cr], axis=2)

    def _ycbcr_to_rgb(self, img_ycbcr):
        """Convert YCbCr to RGB color space"""
        # Ensure input is float32
        img_ycbcr = img_ycbcr.astype(np.float32)
        
        y, cb, cr = img_ycbcr[:,:,0], img_ycbcr[:,:,1], img_ycbcr[:,:,2]
        cb = cb - 0.5
        cr = cr - 0.5
        r = y + 1.402 * cr
        g = y - 0.34414 * cb - 0.71414 * cr
        b = y + 1.772 * cb
        rgb = np.stack([r, g, b], axis=2)
        return np.clip(rgb, 0, 1)

    def _sure_shrink(self, coeffs):
        """SURE (Stein's Unbiased Risk Estimate) thresholding"""
        n = coeffs.size
        squared_coeffs = np.sort(coeffs.flatten()**2)
        c = np.cumsum(squared_coeffs)
        risk = (n - 2 * np.arange(1, n+1) + c) / n
        best_idx = np.argmin(risk)
        threshold = np.sqrt(squared_coeffs[best_idx])
        return threshold

    @lru_cache(maxsize=32)
    def _wavelet_decompose(self, channel_tuple, shape, wavelet, level):
        """Cached wavelet decomposition"""
        # Convert tuple back to array and reshape to original dimensions
        channel = np.array(channel_tuple).reshape(shape)
        return pywt.wavedec2(channel, wavelet, level=level)

    def _enhance_edges(self, original, denoised, strength=0.3):
        """Enhance edges in the denoised image"""
        # Extract high-frequency details from original
        blurred = cv2.GaussianBlur(original, (5, 5), 0)
        high_freq = original - blurred
        
        # Add a portion of these details back to denoised image
        enhanced = denoised + strength * high_freq
        return np.clip(enhanced, 0, 1)

    def denoise_color_image(self, noisy_image, threshold_method='BayesShrink'):
        """Denoise RGB image in YCbCr space"""
        # Convert to YCbCr
        ycbcr = self._rgb_to_ycbcr(noisy_image)
        
        # Apply more aggressive denoising to chroma channels
        y_denoised = self.denoise_image(ycbcr[:,:,0], threshold_method)
        cb_denoised = self.denoise_image(ycbcr[:,:,1], threshold_method, chroma=True)
        cr_denoised = self.denoise_image(ycbcr[:,:,2], threshold_method, chroma=True)
        
        # Stack channels and convert back to RGB
        ycbcr_denoised = np.stack([y_denoised, cb_denoised, cr_denoised], axis=2)
        return self._ycbcr_to_rgb(ycbcr_denoised)

    def denoise_image(self, noisy_image, threshold_method='BayesShrink', chroma=False):
        """
        Denoise image using wavelet thresholding
        
        Parameters:
        noisy_image: Input noisy image
        threshold_method: 'VisuShrink', 'BayesShrink', or 'SureShrink'
        chroma: Whether this is a chroma channel (applies more aggressive denoising)
        
        Returns:
        Denoised image
        """
        # Convert to float for wavelet transform
        img_float = noisy_image.astype(np.float32)
        original_shape = img_float.shape
        
        # Auto-select level if not specified
        if self.level is None:
            self.level = self._get_optimal_level(original_shape)
        
        # Process each channel separately for RGB images
        if img_float.ndim == 3:
            denoised_channels = []
            for c in range(img_float.shape[2]):
                channel = img_float[:,:,c]
                # Pad to avoid boundary effects
                padded_channel = self._pad_to_dyadic(channel)
                # Denoise channel
                denoised_channel = self._denoise_channel(padded_channel, threshold_method, chroma)
                # Crop back to original size
                denoised_channel = denoised_channel[:original_shape[0], :original_shape[1]]
                denoised_channels.append(denoised_channel)
            
            # Combine channels
            denoised_image = np.stack(denoised_channels, axis=2)
        else:
            # Grayscale image
            padded_image = self._pad_to_dyadic(img_float)
            denoised_image = self._denoise_channel(padded_image, threshold_method, chroma)
            # Crop back to original size
            denoised_image = denoised_image[:original_shape[0], :original_shape[1]]
        
        # Enhance edges
        denoised_image = self._enhance_edges(img_float, denoised_image)
        
        # Clip values to [0, 1] range
        denoised_image = np.clip(denoised_image, 0, 1)
        
        return denoised_image

    def _denoise_channel(self, channel, threshold_method, chroma=False):
        """Apply wavelet denoising to a single channel"""
        # Apply wavelet decomposition with caching
        channel_tuple = tuple(channel.flatten())
        shape = channel.shape
        coeffs = self._wavelet_decompose(channel_tuple, shape, self.wavelet, self.level)
        
        # Adjust threshold multiplier for chroma channels
        threshold_multiplier = 2.0 if chroma else 1.0
        
        # Threshold selection based on method
        if threshold_method == 'VisuShrink':
            # VisuShrink (Universal threshold)
            sigma = self._estimate_noise_sigma(coeffs[-1][0])
            threshold = sigma * np.sqrt(2 * np.log(channel.size)) * threshold_multiplier
            
            # Apply thresholding
            new_coeffs = list(coeffs)
            for i in range(1, len(coeffs)):
                new_coeffs[i] = tuple(pywt.threshold(c, threshold, mode='soft') for c in coeffs[i])
                
        elif threshold_method == 'BayesShrink':
            # BayesShrink (adaptive threshold)
            new_coeffs = list(coeffs)
            for i in range(1, len(coeffs)):
                detail_coeffs = []
                for j in range(len(coeffs[i])):
                    sigma = self._estimate_noise_sigma(coeffs[i][j])
                    if sigma == 0:
                        detail_coeffs.append(coeffs[i][j])
                        continue
                    
                    var = np.var(coeffs[i][j])
                    if var <= sigma**2:
                        threshold = float('inf')
                    else:
                        threshold = (sigma**2 / np.sqrt(max(0.001, var - sigma**2))) * threshold_multiplier
                    
                    detail_coeffs.append(pywt.threshold(coeffs[i][j], threshold, mode='soft'))
                
                new_coeffs[i] = tuple(detail_coeffs)
        
        elif threshold_method == 'SureShrink':
            # SureShrink (Stein's Unbiased Risk Estimate)
            new_coeffs = list(coeffs)
            for i in range(1, len(coeffs)):
                detail_coeffs = []
                for j in range(len(coeffs[i])):
                    threshold = self._sure_shrink(coeffs[i][j]) * threshold_multiplier
                    detail_coeffs.append(pywt.threshold(coeffs[i][j], threshold, mode='soft'))
                new_coeffs[i] = tuple(detail_coeffs)
        else:
            # Default: no thresholding, just copy coefficients
            new_coeffs = list(coeffs)
        
        # Reconstruct image
        denoised_channel = pywt.waverec2(new_coeffs, self.wavelet)
        
        return denoised_channel
    
    def _estimate_noise_sigma(self, detail_coeffs):
        """
        Estimate noise standard deviation from detail coefficients
        using MAD (Median Absolute Deviation)
        """
        # MAD estimator for Gaussian noise
        mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
        return mad / 0.6745
    
    def evaluate(self, original, noisy, denoised):
        """Calculate PSNR and SSIM metrics"""
        # Ensure shapes match exactly
        if original.shape != denoised.shape:
            print(f"Warning: Shape mismatch - Original: {original.shape}, Denoised: {denoised.shape}")
            min_shape = [min(original.shape[i], denoised.shape[i]) for i in range(len(original.shape))]
            original = original[:min_shape[0], :min_shape[1], ...] if original.ndim >= 2 else original[:min_shape[0]]
            noisy = noisy[:min_shape[0], :min_shape[1], ...] if noisy.ndim >= 2 else noisy[:min_shape[0]]
            denoised = denoised[:min_shape[0], :min_shape[1], ...] if denoised.ndim >= 2 else denoised[:min_shape[0]]

        # Calculate PSNR
        psnr_noisy = peak_signal_noise_ratio(original, noisy)
        psnr_denoised = peak_signal_noise_ratio(original, denoised)

        # Calculate SSIM with appropriate parameters
        min_dim = min(original.shape[0], original.shape[1])
        # Ensure win_size is odd and <= min_dim
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3

        # Handle multichannel correctly (use channel_axis for new skimage)
        if original.ndim == 3:
            ssim_noisy = structural_similarity(
                original, noisy,
                win_size=win_size,
                data_range=1.0,
                channel_axis=2
            )
            ssim_denoised = structural_similarity(
                original, denoised,
                win_size=win_size,
                data_range=1.0,
                channel_axis=2
            )
        else:
            ssim_noisy = structural_similarity(
                original, noisy,
                win_size=win_size,
                data_range=1.0
            )
            ssim_denoised = structural_similarity(
                original, denoised,
                win_size=win_size,
                data_range=1.0
            )

        return {
            'psnr_noisy': psnr_noisy,
            'psnr_denoised': psnr_denoised,
            'ssim_noisy': ssim_noisy,
            'ssim_denoised': ssim_denoised
        }
    
    def visualize(self, original, noisy, denoised, title=None):
        """Save a visualization of original, noisy, and denoised images side by side"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original, cmap='gray' if original.ndim == 2 else None)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(noisy, cmap='gray' if noisy.ndim == 2 else None)
        axes[1].set_title('Noisy')
        axes[1].axis('off')
        
        axes[2].imshow(denoised, cmap='gray' if denoised.ndim == 2 else None)
        axes[2].set_title('Denoised')
        axes[2].axis('off')
        
        if title:
            fig.suptitle(title)
        plt.tight_layout()
    
        # Save the figure instead of displaying it
        filename = title.replace(" ", "_").lower() if title else "comparison"
        plt.savefig(f'output/{filename}.png')
        plt.close(fig)  # Close the figure to free memory

# Function to create a test image if downloading fails
def create_test_image(size=(512, 512, 3)):
    """Create a test image with various patterns for denoising demonstration"""
    h, w = size[0], size[1]
    
    # Create base patterns
    x, y = np.meshgrid(np.linspace(0, 10, w), np.linspace(0, 10, h))
    
    # Create different patterns for different regions
    img1 = np.sin(x) * np.sin(y)  # Sinusoidal pattern
    img2 = np.zeros((h, w))
    img2[::20, :] = 1  # Horizontal lines
    img2[:, ::20] = 1  # Vertical lines
    
    # Create gradients
    gradient_x = np.tile(np.linspace(0, 1, w), (h, 1))
    gradient_y = np.tile(np.linspace(0, 1, h), (w, 1)).T
    
    # Combine patterns
    regions = np.zeros((h, w))
    regions[:h//2, :w//2] = 0  # Sinusoidal pattern
    regions[:h//2, w//2:] = 1  # Horizontal/vertical lines
    regions[h//2:, :w//2] = 2  # X gradient
    regions[h//2:, w//2:] = 3  # Y gradient
    
    # Apply patterns to regions
    img = np.zeros((h, w))
    img[regions == 0] = img1[regions == 0]
    img[regions == 1] = img2[regions == 1]
    img[regions == 2] = gradient_x[regions == 2]
    img[regions == 3] = gradient_y[regions == 3]
    
    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min())
    
    # Add circular targets in corners for visual reference
    for i, corner in enumerate([(50, 50), (50, w-50), (h-50, 50), (h-50, w-50)]):
        y_c, x_c = corner
        radius = 30
        y_coords, x_coords = np.ogrid[-y_c:h-y_c, -x_c:w-x_c]
        mask = x_coords*x_coords + y_coords*y_coords <= radius*radius
        img[mask] = (i % 2)  # Alternating black and white targets
    
    # Create RGB image
    if len(size) == 3 and size[2] == 3:
        # Create different color channels
        r = img
        g = np.roll(img, shift=img.shape[0]//4, axis=0)
        b = np.roll(img, shift=img.shape[1]//4, axis=1)
        img_rgb = np.stack([r, g, b], axis=2)
        return img_rgb
    else:
        return img

# Helper function to download or generate Kodak dataset images
def get_kodak_image(image_number):
    """Download an image from the Kodak dataset or return a test image"""
    url = f"https://r0k.us/graphics/kodak/kodak/kodim{image_number:02d}.png"
    local_dir = "kodak_dataset"
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, f"kodim{image_number:02d}.png")

    # Check if image exists locally
    if os.path.exists(local_path):
        try:
            img = Image.open(local_path).convert("RGB")
            return np.array(img) / 255.0
        except Exception as e:
            print(f"Failed to open local image: {e}")

    # Try downloading the image
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return np.array(img) / 255.0
        else:
            print(f"Failed to download image: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading image: {e}")

    # Fallback to test image
    print(f"Using fallback test image for kodim{image_number:02d}")
    return create_test_image()

# Main function to run the demo
def main():
    # Create output directory
    os.makedirs('output', exist_ok=True)
    os.makedirs('kodak_dataset', exist_ok=True)
    
    # Initialize denoiser
    denoiser = WaveletDenoiser(wavelet='bior4.4', level=2)
    
    # Process images
    for img_num in [1, 5, 9, 23]:  # Selected a few images
        print(f"Processing Kodak image {img_num}...")
        
        # Get image
        original_img = get_kodak_image(img_num)
        
        # Make sure image is large enough (at least 32x32 for level=2)
        if original_img.shape[0] < 32 or original_img.shape[1] < 32:
            print(f"Image too small, resizing to minimum dimensions")
            if original_img.ndim == 3:
                original_img = cv2.resize(original_img, (max(32, original_img.shape[1]), max(32, original_img.shape[0])))
            else:
                original_img = cv2.resize(original_img, (max(32, original_img.shape[1]), max(32, original_img.shape[0])))
        
        # Add noise
        noise_level = 0.02  # Adjust noise level as needed
        noisy_img = denoiser.add_gaussian_noise(original_img, var=noise_level)
        
        # Denoise image
        denoised_img = denoiser.denoise_image(noisy_img, threshold_method='BayesShrink')
        
        # Make sure all images have same shape before evaluation
        min_shape = [min(s1, s2, s3) for s1, s2, s3 in zip(original_img.shape, noisy_img.shape, denoised_img.shape)]
        original_img_crop = original_img[:min_shape[0], :min_shape[1], ...] if len(min_shape) > 2 else original_img[:min_shape[0], :min_shape[1]]
        noisy_img_crop = noisy_img[:min_shape[0], :min_shape[1], ...] if len(min_shape) > 2 else noisy_img[:min_shape[0], :min_shape[1]]
        denoised_img_crop = denoised_img[:min_shape[0], :min_shape[1], ...] if len(min_shape) > 2 else denoised_img[:min_shape[0], :min_shape[1]]
        
        # Evaluate results
        try:
            metrics = denoiser.evaluate(original_img_crop, noisy_img_crop, denoised_img_crop)
            print(f"Image {img_num} PSNR: Noisy = {metrics['psnr_noisy']:.2f}dB, Denoised = {metrics['psnr_denoised']:.2f}dB")
            print(f"Image {img_num} SSIM: Noisy = {metrics['ssim_noisy']:.4f}, Denoised = {metrics['ssim_denoised']:.4f}")
        except Exception as e:
            print(f"Error evaluating metrics: {e}")
            print("Continuing with visualization...")
        
        # Visualize results
        denoiser.visualize(original_img, noisy_img, denoised_img, f"Kodak Image {img_num}")
        
        # Save results
        plt.imsave(f'output/original_{img_num}.png', np.clip(original_img, 0, 1))
        plt.imsave(f'output/noisy_{img_num}.png', np.clip(noisy_img, 0, 1))
        plt.imsave(f'output/denoised_{img_num}.png', np.clip(denoised_img, 0, 1))

if __name__ == "__main__":
    main()