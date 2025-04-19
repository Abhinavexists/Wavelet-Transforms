import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from transform import WaveletDenoiser, get_kodak_image
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
from concurrent.futures import ProcessPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser(description="Wavelet-based image denoising")
    parser.add_argument("--images", type=int, nargs='+', default=[1, 5, 9, 23],
                        help="Image numbers from Kodak dataset")
    parser.add_argument("--noise", type=float, default=0.02,
                        help="Noise level (variance for Gaussian noise)")
    parser.add_argument("--wavelet", type=str, default="bior4.4",
                        choices=["haar", "db4", "sym4", "coif3", "bior4.4"],
                        help="Wavelet type")
    parser.add_argument("--level", type=int, default=None,
                        help="Decomposition level (default: auto-selected)")
    parser.add_argument("--method", type=str, default="BayesShrink",
                        choices=["VisuShrink", "BayesShrink", "SureShrink"],
                        help="Thresholding method")
    parser.add_argument("--color-space", type=str, default="RGB",
                        choices=["RGB", "YCbCr"],
                        help="Color space for processing")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory for results")
    return parser.parse_args()

def compare_methods(image, noise_level=0.02):
    """Compare different denoising methods on the same image"""
    # Original and noisy
    noisy = random_noise(image, mode='gaussian', var=noise_level)
    
    # Different wavelets
    wavelets = ['haar', 'db4', 'sym4', 'bior4.4']
    results = {'original': image, 'noisy': noisy}
    metrics = {}
    
    for wavelet in wavelets:
        denoiser = WaveletDenoiser(wavelet=wavelet)
        denoised = denoiser.denoise_image(noisy)
        results[f'wavelet_{wavelet}'] = denoised
        metrics[f'wavelet_{wavelet}'] = {
            'psnr': peak_signal_noise_ratio(image, denoised),
            'ssim': structural_similarity(image, denoised, 
                      data_range=1.0, channel_axis=2 if image.ndim==3 else None)
        }
    
    # Compare with non-wavelet methods
    # Bilateral Filter
    if image.ndim == 3:
        # Process each channel for color images
        bilateral = np.zeros_like(image)
        for c in range(image.shape[2]):
            bilateral[:,:,c] = cv2.bilateralFilter(
                (noisy[:,:,c] * 255).astype(np.uint8), d=5, sigmaColor=75, sigmaSpace=75) / 255.0
    else:
        bilateral = cv2.bilateralFilter(
            (noisy * 255).astype(np.uint8), d=5, sigmaColor=75, sigmaSpace=75) / 255.0
    
    results['bilateral'] = bilateral
    metrics['bilateral'] = {
        'psnr': peak_signal_noise_ratio(image, bilateral),
        'ssim': structural_similarity(image, bilateral, 
                 data_range=1.0, channel_axis=2 if image.ndim==3 else None)
    }
    
    return results, metrics

def plot_comparison(results, metrics, output_dir, image_num):
    """Plot and save comparison of different methods"""
    n_methods = len(results)
    fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (method, img) in enumerate(results.items()):
        ax = axes[idx]
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        title = method.replace('_', ' ').title()
        if method in metrics:
            title += f'\nPSNR: {metrics[method]["psnr"]:.2f}\nSSIM: {metrics[method]["ssim"]:.3f}'
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_{image_num}.png'))
    plt.close()

def process_single_image(args, image_num):
    """Process a single image with the specified parameters"""
    # Get image
    image = get_kodak_image(image_num)
    if image is None:
        print(f"Failed to load image {image_num}")
        return None
    
    # Add noise
    noisy = random_noise(image, mode='gaussian', var=args.noise)
    
    # Create denoiser
    denoiser = WaveletDenoiser(wavelet=args.wavelet, level=args.level)
    
    # Denoise image
    if args.color_space == 'YCbCr':
        denoised = denoiser.denoise_color_image(noisy, args.method)
    else:
        denoised = denoiser.denoise_image(noisy, args.method)
    
    # Compare methods
    results, metrics = compare_methods(image, args.noise)
    results['wavelet_denoised'] = denoised
    metrics['wavelet_denoised'] = {
        'psnr': peak_signal_noise_ratio(image, denoised),
        'ssim': structural_similarity(image, denoised, 
                 data_range=1.0, channel_axis=2 if image.ndim==3 else None)
    }
    
    # Plot and save results
    plot_comparison(results, metrics, args.output_dir, image_num)
    
    return metrics

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process images in parallel
    with ProcessPoolExecutor() as executor:
        futures = []
        for img_num in args.images:
            futures.append(
                executor.submit(process_single_image, args, img_num)
            )
        
        # Collect results
        all_metrics = {}
        for img_num, future in zip(args.images, futures):
            metrics = future.result()
            if metrics:
                all_metrics[img_num] = metrics
        
        # Print summary
        print("\nResults Summary:")
        print("-" * 80)
        for img_num, metrics in all_metrics.items():
            print(f"\nImage {img_num}:")
            for method, method_metrics in metrics.items():
                print(f"  {method}:")
                print(f"    PSNR: {method_metrics['psnr']:.2f}")
                print(f"    SSIM: {method_metrics['ssim']:.3f}")

if __name__ == "__main__":
    main() 