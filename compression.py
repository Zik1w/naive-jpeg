from PIL import Image
import numpy as np
import math
import gzip
import bz2
import lzma
import os


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def crop(image,block_size):
    if image.shape[0] % block_size != 0:
        cropped_width = math.floor(int(image.shape[0] / block_size) * block_size)
    else:
        cropped_width = image.shape[0]

    if image.shape[1] % block_size != 0:
        cropped_height = math.floor(int(image.shape[1] / block_size) * block_size)
    else:
        cropped_height = image.shape[1]

    cropped_image = np.zeros((cropped_width, cropped_height))

    for rownum in range(len(cropped_image)):
       for colnum in range(len(cropped_image[rownum])):
           cropped_image[rownum][colnum] = image[rownum][colnum]

    return cropped_image


def partition(image,block_size):
    data = np.split(image, image.shape[0] / block_size)
    res = []

    for arr in data:
        res.append(np.split(arr, arr.shape[1] / block_size, axis=1))

    return res


def dctmtx(block_size):
    dct_matrix = np.zeros((block_size, block_size))

    for i in range(0, block_size):
        for j in range(0, block_size):
            if i == 0:
                dct_matrix[i][j] = 1/math.sqrt(block_size)
            else:
                dct_matrix[i][j] = math.sqrt(2/block_size) * math.cos((math.pi*(2 * j + 1) * i)/(2*block_size))

    return dct_matrix


def dct(input_matrix, dct_mtx):
    dct_matrix = np.matmul(np.matmul(dct_mtx,input_matrix), dct_mtx.transpose())
    return dct_matrix


def quantize(input_matrix,block_size,qr):

    quantize_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]])

    if block_size == 16:
        quantize_matrix_double = np.zeros((block_size, block_size))
        for i in range(0,8):
            for j in range(0, 8):
                temp = quantize_matrix[i][j]
                quantize_matrix_double[2 * i][2 * j] = temp
                quantize_matrix_double[2 * i + 1][2 * j] = temp
                quantize_matrix_double[2 * i][2 * j + 1] = temp
                quantize_matrix_double[2 * i + 1][2 * j + 1] = temp
        used_quantize_matrix = quantize_matrix_double
    else:
        used_quantize_matrix = quantize_matrix

    used_quantize_matrix = used_quantize_matrix/qr
    quantized_matrix = input_matrix/used_quantize_matrix
    quantized_matrix = np.rint(quantized_matrix)
    return quantized_matrix


def zig_zag(input_matrix,block_size):
    z = np.empty([block_size*block_size])
    index = -1
    bound = 0
    for i in range(0, 2 * block_size -1):
        if i < block_size:
            bound = 0
        else:
            bound = i - block_size + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                z[index] = input_matrix[j, i-j]
            else:
                z[index] = input_matrix[i-j, j]
    return z


def gzip_comp(output,output_file):
    filename = output_file + ".txt.gz"
    with gzip.open(filename, "wb") as f:
        f.write(output)
    print("size for %s is %s bytes" % (filename, os.path.getsize(filename)))
    return filename


def lzma_comp(output,output_file):
    filename = output_file + ".xz"
    with lzma.open(filename, "w") as f:
        f.write(output)
    print("size for %s is %s bytes" % (filename, os.path.getsize(filename)))
    return filename

def bz2_comp(output,output_file):
    filename = output_file + ".bz2"
    with bz2.open(filename, "wb") as f:
        f.write(output)
    print("size for %s is %s bytes" % (filename, os.path.getsize(filename)))
    return filename


def main(image_file, block_size, qr, binary_length, compression_method):
    im = Image.open(image_file)
    im_matrix = crop(np.array(im), block_size)
    res = partition(im_matrix, block_size)

    result_string = '';
    result_string += "{0:b}".format(im_matrix.shape[0]) + "x"
    result_string += "{0:b}".format(im_matrix.shape[1]) + "\n"

    dct_mtx = dctmtx(block_size)

    for i in range(len(res)):
        for j in range(len(res[i])):
            dct_matrix = dct(res[i][j], dct_mtx)
            quantize_matrix = quantize(dct_matrix, block_size, qr)

            # represent DC coefficients by difference
            if i == 0 and j == 0:
                temp_difference = quantize_matrix[0][0]
            else:
                quantize_matrix[0][0] = quantize_matrix[0][0] - temp_difference

            temp_difference = quantize_matrix[0][0]

            binary_string = ''.join(('{0:' + str(binary_length) + 'b}').format(int(x)) for x in zig_zag(quantize_matrix, block_size))
            result_string += binary_string + '\n'

    result_bytes = str.encode(result_string)

    name_list = image_file.split(".")

    output_file = name_list[0] + "_" + str(block_size) + "_" + str(qr) + "_" + str(bl)

    if compression_method == "gzip":
        gzip_comp(result_bytes, output_file)
    elif compression_method == "bzip":
        bz2_comp(result_bytes, output_file)
    elif compression_method == "xz":
        lzma_comp(result_bytes, output_file)
    else:
        print("the compression method is not found!")


if __name__ == "__main__":
    file_name = input('Enter file name: ')
    bs = int(input('Enter block size: '))
    quantize_ratio = int(input('Enter quantize ratio: '))
    bl = int(input('Enter binary length: '))
    compression_method = input('Enter lossless compression method: ')
    main(file_name, bs, quantize_ratio, bl, compression_method)
