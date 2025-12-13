import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim

from decoder import BaselineJPEGDecoder
# from reference_decode import decode_reference_y
# from metrics import compute_psnr, compute_ssim

import argparse

def decode_reference_y(jpeg_path: str) -> np.ndarray:
    bgr = cv2.imread(jpeg_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to read image")

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0]
    return Y


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    assert img1.shape == img2.shape
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    mse = np.mean(diff * diff)
    if mse == 0:
        return float('inf')
    return 10 * math.log10((255 * 255) / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    score, _ = ssim(img1, img2, full=True)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify JPEG Decoder with OpenCV')
    parser.add_argument('--ori-img', '-or', required=True, type=str,
                        help='path of original JPEG image')
    parser.add_argument('--we-implement-decoded', '-wid', required=True, type=str,
                        help='path of decoded Y raw image')
    args = parser.parse_args()

    # Reference decoder
    ref_y = decode_reference_y(args.ori_img)
    h, w = ref_y.shape

    # Read your decoded Y
    my_y = np.fromfile(args.we_implement_decoded, dtype=np.uint8)
    my_y = my_y.reshape((h, w))

    print("Compare image(Y) we decoded with OpenCV decoding")
    print("PSNR (Y):", compute_psnr(ref_y, my_y))
    print("SSIM (Y):", compute_ssim(ref_y, my_y))
