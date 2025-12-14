## Overview

We implement a baseline-jpeg-decoder in python. It will read the image and generate the raw file of the image.<br>
We then compare the output images we decode by ourselves with ones OpenCV decodes in Y(luminance) channel.<br>
We evaluate pairs of output images with PSNR and SSIM.<br>

## How to run it

1.install requirements<br>
```
pip install -r requirements.txt
```
2.get your input or use ours<br>

place your .jpeg or .jpg in the inputs folder (please make sure the image is in baseline not progressive)<br>

3.Run mainn.py<br>
```
python3 main.py dog.jpg
```

4.the result<br>

the result will first show in a window after you close it then the verify result will show out<br>
there will be a raw file in the outputs <br>

## code sturcture
```
vc-final/
│
├── inputs/
│   ├── car.jpeg                          # Sample input image 1
│   └── cat1.jpg                          # Sample input image 2
│   └── dog.jpg                           # Sample input image 3
│
├── outputs/
│   ├── decoded_car.raw                   # Output: .raw for car.jpeg
│   └── decoded_cat1.raw                  # Output: .raw for cat1.jpg
│   └── decoded_dog.raw                   # Output: .raw for dog.jpg
│
├── pics/
│   ├── decode_car.png                    # Sample metrics output
│   ├── decode_cat1.png                   # Sample metrics output
│   ├── decode_dog.png                    # Sample metrics output
│   ├── verify_with_cv2_car.png           # Sample metrics output
│   ├── verify_with_cv2_car.png           # Sample metrics output
│   ├── verify_with_cv2_car.png           # Sample metrics output
│
├── decoder.py                            # Baseline JPEG decoder implementation
├── codec.py                              # Zigzag pattern generator 
├── main.py                               # Deal with the input and combine with the decoder
├── verify.py                             # Verifies decoded image accuracy using Pillow and outputs metrics
├── verify_with_cv2_decoder.py            # Verifies decoded image accuracy using OpenCV decodes in Y(luminance) channel and outputs metrics
│
├── requirements.txt                      # List of required Python libraries

```

