import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_and_compare_images(your_raw_path, reference_jpeg_path):
    """
    Load and compare your decoded raw file with a reference decoder (Pillow)
    
    Args:
        your_raw_path: Path to your decoded raw file
        width: Width of the image
        height: Height of the image
        reference_jpeg_path: Path to the original JPEG file
    """
    # Load and decode reference image using Pillow
    reference_image = np.array(Image.open(reference_jpeg_path))

    # Load your raw file
    with open(your_raw_path, 'rb') as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
    your_image = raw_data.reshape(reference_image.shape)
    
    
    
    # Ensure both images have the same shape
    if your_image.shape != reference_image.shape:
        raise ValueError(f"Image shapes don't match: {your_image.shape} vs {reference_image.shape}")
    
    # Calculate various metrics
    results = {}
    
    # 1. Mean Absolute Error (MAE)
    mae = np.mean(np.abs(your_image - reference_image))
    results['MAE'] = mae
    
    # 2. Mean Squared Error (MSE)
    mse = np.mean((your_image - reference_image) ** 2)
    results['MSE'] = mse
    
    # 3. Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    results['RMSE'] = rmse
    
    # 4. Peak Signal-to-Noise Ratio (PSNR)
    # Higher is better, typical values are between 30 and 50 dB
    results['PSNR'] = psnr(reference_image, your_image)
    
    # 5. Structural Similarity Index (SSIM)
    # Ranges from -1 to 1, where 1 means identical images
    results['SSIM'] = ssim(reference_image, your_image, channel_axis=2)
    
    # 6. Per-channel differences
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        channel_mae = np.mean(np.abs(your_image[:,:,i] - reference_image[:,:,i]))
        results[f'{channel} Channel MAE'] = channel_mae
    
    return results

def print_results(results):
    """Print the comparison results in a formatted way"""
    print("\nImage Comparison Metrics:")
    print("-" * 50)
    for metric, value in results.items():
        print(f"{metric:20s}: {value:.6f}")


def verify_decoder(jpeg_path, your_raw_path):
    """Main function to verify decoder output"""
    try:
        results = load_and_compare_images(your_raw_path, jpeg_path)
        print_results(results)
        
        # Provide interpretation of results
        if results['PSNR'] > 40:
            print("\nPSNR is excellent (>40 dB)")
        elif results['PSNR'] > 30:
            print("\nPSNR is good (>30 dB)")
        else:
            print("\nPSNR is poor (<30 dB)")
            
        if results['SSIM'] > 0.95:
            print("SSIM indicates very high similarity")
        elif results['SSIM'] > 0.90:
            print("SSIM indicates good similarity")
        else:
            print("SSIM indicates potential issues")
            
    except Exception as e:
        print(f"Error during verification: {str(e)}")