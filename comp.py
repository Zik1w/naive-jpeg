from PIL import Image
import numpy as np
import math
import gzip
import bz2
import psnr
import lzma
import os


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


def zig_zag_reverse(input_matrix,block_size):
    output_matrix = np.empty([block_size,block_size])
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
                output_matrix[j, i - j] = input_matrix[index]
            else:
                output_matrix[i - j, j] = input_matrix[index]
    return output_matrix


def quantize_back(input_matrix,block_size,qr):

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
    quantized_matrix = np.multiply(input_matrix,used_quantize_matrix)
    quantized_matrix = np.rint(quantized_matrix)

    return quantized_matrix


def idct(input_matrix, dct_mtx):
    idct_matrix = np.matmul(np.matmul(dct_mtx.transpose(),input_matrix), dct_mtx)
    return idct_matrix


def gzip_comp(output,output_file):
    filename = output_file + ".txt.gz"
    with gzip.open(filename, "wb") as f:
        f.write(output)
    # print(os.path.getsize(filename))
    return filename


def lzma_comp(output,output_file):
    filename = output_file + ".xz"
    with lzma.open(filename, "w") as f:
        f.write(output)
    # print(os.path.getsize(filename))
    return filename


def bz2_comp(output,output_file):
    filename = output_file + ".bz2"
    with bz2.open(filename, "wb") as f:
        f.write(output)
    # print(os.path.getsize(filename))
    return filename


def gzip_decomp(file):
    with gzip.open(file, "rb") as f:
        f_content = f.read()
    return f_content


def bz2_decomp(file):
    with bz2.open(file, "rb") as f:
        f_content = f.read()
    return f_content


def lzma_decomp(file):
    with lzma.open(file) as f:
        f_content = f.read()
    return f_content


def main(image_file,block_size,qr,binary_length,compression_method):
    im = Image.open(image_file)
    im_matrix = crop(np.array(im),block_size)
    res = partition(im_matrix,block_size)

    result_string = '';
    result_string += "{0:b}".format(im_matrix.shape[0]) + "x"
    result_string += "{0:b}".format(im_matrix.shape[1]) + "\n"

    dct_mtx = dctmtx(block_size)

    for i in range(len(res)):
        for j in range(len(res[i])):
            dct_matrix = dct(res[i][j], dct_mtx)
            quantize_matrix = quantize(dct_matrix,block_size,qr)

            #represent DC coefficients by difference
            if i == 0 and j == 0:
                temp_difference = quantize_matrix[0][0]
            else:
                quantize_matrix[0][0] = quantize_matrix[0][0] - temp_difference

            temp_difference = quantize_matrix[0][0]
            # print(temp_difference)
            binary_string = ''.join(('{0:' + str(binary_length) + 'b}').format(int(x)) for x in zig_zag(quantize_matrix, block_size))
            # print(binary_string)
            result_string += binary_string + '\n'

    result_bytes = str.encode(result_string)

    name_list = image_file.split(".")
    output_file = name_list[0] + "_" + str(block_size) + "_" + str(qr) + "_" + str(binary_length)

    filename = "";

    #decompress
    if compression_method == "gzip":
        filename = gzip_comp(result_bytes, output_file)
    elif compression_method == "bzip":
        filename = bz2_comp(result_bytes, output_file)
    elif compression_method == "xz":
        filename = lzma_comp(result_bytes, output_file)
    else:
        print("the compression method is not found!")

    #decompress
    if filename.lower().endswith(".gz"):
        result_bytes = gzip_decomp(filename)
    elif filename.lower().endswith(".bz2"):
        result_bytes = bz2_decomp(filename)
    elif filename.lower().endswith(".xz"):
        result_bytes = lzma_decomp(filename)

    b_string = result_bytes.decode()

    b_array = b_string.split('\n')

    sz = b_array[0];
    image_size_info = sz.split("x")
    height = int(image_size_info[0], 2)
    width = int(image_size_info[1], 2)

    output_matrix = np.zeros((height, width))

    height_size = int(height / block_size)
    wide_size = int(width / block_size)

    temp_restore = 0
    for i in range(1, len(b_array) - 1):
        temp_string = b_array[i]
        numbers = [int(temp_string[i:i + binary_length], 2) for i in range(0, len(temp_string), binary_length)]
        b_z = zig_zag_reverse(numbers, block_size)

        # restore DC coefficients
        if i == 1:
            zero_p = b_z[0][0]
        elif i == 2:
            temp_restore = b_z[0][0]
            b_z[0][0] = b_z[0][0] + zero_p
        elif i > 2:
            temp_restore = b_z[0][0]
            b_z[0][0] = b_z[0][0] + dc_coefficient

        dc_coefficient = temp_restore

        qb_matrix = quantize_back(b_z,block_size,qr)
        idct_matrix = idct(qb_matrix, dct_mtx)

        m = int(math.floor((i - 1) / wide_size))  # height
        n = int((i - 1) % wide_size)  # width

        x = block_size * m
        for p in range(len(idct_matrix)):
            y = block_size * n
            for q in range(len(idct_matrix[p])):
                # print("x = %d, y = %d" % (x, y))
                output_matrix[x][y] = idct_matrix[p][q]
                y += 1
            x += 1

    res_db = psnr.compare_array(output_matrix,image_file)
    # print("PSNR is %d db for %s" % (res_db,image_file))
    # new_im = Image.fromarray(output_matrix)
    # new_im.show()

if __name__ == "__main__":
    bs = int(input('Enter block size: '))
    quantize_ratio = int(input('Enter quantize ratio: '))
    bl = int(input('Enter binary length: '))
    cm1 = input('Enter 1st lossless compression method: ')
    main("Kodak08gray.bmp", bs, quantize_ratio, bl, cm1)
    main("Kodak09gray.bmp", bs, quantize_ratio, bl, cm1)
    main("Kodak12gray.bmp", bs, quantize_ratio, bl, cm1)
    main("Kodak18gray.bmp", bs, quantize_ratio, bl, cm1)
    main("Kodak21gray.bmp", bs, quantize_ratio, bl, cm1)
    main("Kodak22gray.bmp", bs, quantize_ratio, bl, cm1)
    cm2 = input('Enter 2nd lossless compression method: ')
    main("Kodak08gray.bmp", bs, quantize_ratio, bl, cm2)
    main("Kodak09gray.bmp", bs, quantize_ratio, bl, cm2)
    main("Kodak12gray.bmp", bs, quantize_ratio, bl, cm2)
    main("Kodak18gray.bmp", bs, quantize_ratio, bl, cm2)
    main("Kodak21gray.bmp", bs, quantize_ratio, bl, cm2)
    main("Kodak22gray.bmp", bs, quantize_ratio, bl, cm2)
    cm3 = input('Enter 3rd lossless compression method: ')
    main("Kodak08gray.bmp", bs, quantize_ratio, bl, cm3)
    main("Kodak09gray.bmp", bs, quantize_ratio, bl, cm3)
    main("Kodak12gray.bmp", bs, quantize_ratio, bl, cm3)
    main("Kodak18gray.bmp", bs, quantize_ratio, bl, cm3)
    main("Kodak21gray.bmp", bs, quantize_ratio, bl, cm3)
    main("Kodak22gray.bmp", bs, quantize_ratio, bl, cm3)
