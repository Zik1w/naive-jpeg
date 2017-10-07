# JPEG

A simply implementation of the JPEG algorithm in Python Based on the Study Material from CS175.

**compression.py**: Perform a series of steps in order to compress the image file: 1)crop the image 2) subtract 128 3)partition into block 4) 2d-dct with dat matrix 5)quantization 6) calculate dc difference 7)zigzag the array and convert the 1d array to binary string representation 8)use lossless compression method to compress and save the compressed file

**decompression.py**: Perform  a series of steps in order to decompress the image file: 1)use lossless compression method to decompress the file 2) read the raw binary string use reverse zigzag 3) restore dc coefficients 4) multiple the array by quantization matrix 5) use inverse 2d-dct 6)put the resulting blocks into the corresponding result array 7) output the image file


**psnr.py**: Perform the PSNR measurement for the input files using the standard formula, it can also be called by other program and use its compare_array() function directly to calculate the PNSR values between an 2d array and an image.


**comp.py**: file is an integrated file containing both compression and decompression. It is also a simply way to call both all three lossless compression algorithms on all six test files in a more consistent and combined way. Then perform psnr calculation automatically.

