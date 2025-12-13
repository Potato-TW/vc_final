from decoder import BaselineJPEGDecoder
import argparse
import time
import matplotlib.pyplot as plt
from pathlib import Path
import verify # Assuming verify exists as in original upload

def decode_image(filename: str):
    # Use Pathlib for robust path handling
    base_dir = Path(__file__).parent
    input_path = base_dir / 'inputs' / filename
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"\nProcessing: {input_path.name}")

    output_dir = base_dir / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f'decoded_{input_path.stem}'
    output_path_rgb = f'{output_path}_rgb.raw'
    output_path_y = f'{output_path}_y.raw'

    decoder = BaselineJPEGDecoder(str(input_path))

    try:
        start_time = time.time()
        decoded_image_rgb, decoded_image_y = decoder.decode()
        duration = time.time() - start_time
        
        print(f"Decoding time: {duration:.4f} seconds")
        print(f"Image Shape: {decoded_image_rgb.shape}")

        # Display
        plt.figure(figsize=(10, 8))
        plt.imshow(decoded_image_rgb)
        plt.axis('off')
        plt.title(f"Decoded: {filename}")
        plt.show()

        # Save Raw
        decoded_image_rgb.tofile(output_path_rgb)
        decoded_image_y.tofile(output_path_y)
        print(f"Successfully decoded to: {output_path}")
        
        # Verification (wrapped in try block to prevent crash if verify module issues exist)
        try:
            verify.verify_decoder(str(input_path), str(output_path_rgb))
        except Exception as v_err:
            print(f"Verification warning: {v_err}")
            
    except Exception as e:
        print(f"Error during decoding: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Optimized JPEG Decoder')
    parser.add_argument('image', type=str, help='Name of input JPEG image (e.g., test.jpg)')
    args = parser.parse_args()

    decode_image(args.image)

if __name__ == "__main__":
    main()