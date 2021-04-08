import cv2
import numpy as np
from matplotlib import pyplot as plt

eightBitRGB=[[0., 0., 0.],
             [0., 0., 1.],
             [0., 1., 0.],
             [0., 1., 1.],
             [1., 0., 0.],
             [1., 0., 1.],
             [1., 1., 0.],
             [1., 1., 1.]]

sixteenBitRGB= [[0, 0, 0],
                [0, 1, 1],
                [0, 0, 1],
                [1, 0, 1],
                [0, 0.5, 0],
                [0.5, 0.5, 0.5],
                [0, 1, 0],
                [0.5, 0, 0],
                [0, 0, 0.5],
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [1, 0, 0],
                [0.75, 0.75, 0.75],
                [0, 0.5, 0.5],
                [1, 1, 1],
                [1, 1, 0]]

twoBit = [[0],
          [1]]

fourBit = [[0],
           [0.33],
           [0.66],
           [1]]

twoBit = np.array(twoBit)
fourBit = np.array(fourBit)
eightBitRGB = np.array(eightBitRGB)
sixteenBitRGB = np.array(sixteenBitRGB)


def colorfit(color, palette):
    return palette[np.argmin(np.linalg.norm(palette - color, axis=1))].astype(float)


def dithering_random(image):
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            r = np.random.rand()
            if image[y, x] > r:
                image[y, x] = 1.0
            elif image[y, x] <= r:
                image[y, x] = 0.0

    return image


def counting_bits(scr_img):
    bitCount = np.count_nonzero(scr_img.shape)
    if bitCount == 3 and isinstance(scr_img[0][0][0], np.float32):
        hei, wid, cana = scr_img.shape
    elif bitCount == 3 and isinstance(scr_img[0][0][0], np.uint8):
        scr_img = scr_img.astype(float)
        scr_img = scr_img / 255
        hei, wid, cana = scr_img.shape
    elif bitCount == 2 and isinstance(scr_img[0][0], np.uint8):
        scr_img = scr_img.astype(float)
        scr_img = scr_img / 255
        hei, wid = scr_img.shape

    return hei, wid, scr_img


def ordered_dethering(scr_img, palette):
    hei, wid, can = scr_img.shape
    m2 = [[0.0, 8.0, 2.0, 4.0],
          [12.0, 4.0, 14.0, 6.0],
          [3.0, 11.0, 1.0, 9.0],
          [15.0, 7.0, 13.0, 5.0]]
    m2 = np.array(m2)
    m2 = (1/16) * (m2 + 1) - 0.5

    bits = 4
    image = np.copy(scr_img)
    for y in range(hei):
        for x in range(wid):
            image[y,x] = colorfit(image[y,x] + m2[y % bits, x % bits], palette)

    return image


def floyd_steinberg_dithering(scr_img, palette):
    hei, wid, can = scr_img.shape
    image = np.copy(scr_img)
    for y in range(hei-1):
        for x in range(wid-1):
            pix = image[y][x]
            newpix = colorfit(pix, palette)

            error = pix - newpix

            image[y][x] = newpix

            image[y    ][x + 1] += error * 7/16
            image[y + 1][x - 1] += error * 3/16
            image[y + 1][x    ] += error * 5/16
            image[y + 1][x + 1] += error * 1/16

    return image


scr_name = "0009.png"
scr_img = cv2.imread(scr_name)
scr_img = np.flip(scr_img, 2)
palette = fourBit
grey = True


plt.subplot(1, 5, 1)
plt.title("Original")
plt.imshow(scr_img)

hei, wid, scr_img = counting_bits(scr_img)

img = np.copy(scr_img)
for y in range(hei):
    for x in range(wid):
      img[y, x] = colorfit(img[y, x], palette)


plt.subplot(1, 5, 2)
plt.title("Color Fitting")
plt.imshow(img)

if grey:
    result = img[:, :, 0]
    random = dithering_random(result)
    plt.subplot(1, 5, 3)
    plt.title("Random Dithering")
    plt.imshow(random, cmap="gray")

ordered = ordered_dethering(scr_img, palette)
plt.subplot(1, 5, 4)
plt.title("Ordered Dithering")
plt.imshow(ordered)

floyd = floyd_steinberg_dithering(scr_img, palette)
plt.subplot(1, 5, 5)
plt.title("Floyd–Steinberg Dithering")
plt.imshow(floyd)


plt.show()
