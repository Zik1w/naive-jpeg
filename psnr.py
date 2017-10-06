from PIL import Image
import numpy as np
import math


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


def psnr(compressed_image,origin_image):

    compressed = np.array(Image.open(compressed_image))
    origin = np.array(Image.open(origin_image))
    mse = 0

    for rownum in range(len(compressed)):
       for colnum in range(len(compressed[rownum])):
           mse += math.pow((origin[rownum][colnum] - compressed[rownum][colnum]),2)

       # print(mse)

    mse = mse/(len(origin)*len(origin[0]))

    res = 10 * math.log10(math.pow(255,2)/mse)

    return res


def compare_array(array,origin_image):

    origin = np.array(Image.open(origin_image))
    mse = 0

    for rownum in range(len(array)):
       for colnum in range(len(array[rownum])):
           mse += math.pow((origin[rownum][colnum] - array[rownum][colnum]),2)

       # print(mse)

    mse = mse/(len(origin)*len(origin[0]))

    res = 10 * math.log10(math.pow(255,2)/mse)

    return res


def main(compressed,origin):
    # origin = input("Enter origin file name: ")
    # compressed = input("Enter compressed file name: ")

    res = psnr(compressed,origin)

    print("%d db" % res)

if __name__ == "__main__":
    main("kodak09gray8gz1.jpeg", "Kodak09gray.bmp")