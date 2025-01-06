import math
import os
import tkinter as tk

import cv2 as cv
import numpy as np

"""
This script is designed to draw the logo of the company vEMPIRE
"""

##### System parameters to center the resulting images

root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

##### Defining the desired window width/height and computing the resulting positions for the grid

offset = 30
window_height = 400
window_width = 400
grid_height = 2 * (window_height + offset) - offset
grid_width = 3 * window_width
x_offset = (screen_width - grid_width) // 2
y_offset = (screen_height - grid_height) // 2

#### We define some parameters such as the colors to use in the "wings" or the contour color/size

color_red = (50,4,250)
color_red_shadow = (41,7,161)
color_contour = (0,0,0)
contour_size = 8
resolution = 1080

def bernstein_polynomial(points : np.array, t : float):

    """
    This function is used to compute the bernstein polynomial for points evaluated at t
    """

    n = points.shape[0]
    values = [math.comb(n - 1, i) * ((1 - t) ** (n - 1 - i)) * (t ** i) * points[i] for i in range(n)]
    return np.sum(values, axis = 0)

def bezier_curve(points : np.array, precision : int = 1000):

    """
    This function is used to generate precision points for the bezier curve with the points given
    """

    return np.array([bernstein_polynomial(points, t) for t in np.linspace(0, 1, precision)], dtype = np.int32)

def draw_logo(color_blank : np.uint8, 
              color_contour : tuple[np.uint8, np.uint8, np.uint8], 
              color_fang : tuple[np.uint8, np.uint8, np.uint8], 
              color_red : tuple[np.uint8, np.uint8, np.uint8], 
              color_red_shadow : tuple[np.uint8, np.uint8, np.uint8], 
              contour : bool, 
              contour_size : int):
    
    """
    This function is used to draw the logo based on different parameters
    """

    ##### We start with a blank image, may it be white or dark, we use a very large image at the beginning to allow anti-aliasing

    image = np.zeros((resolution * 4 , resolution * 4, 3), np.uint8) + color_blank

    ##### The first points are for the left left bezier curve

    points_1 = np.array([[225 - 5, 300],
                         [265 - 5, 305],
                         [410 - 5, 450],
                        [ 250 - 5, 545]], dtype = np.int32)
    points_1 = 4 * points_1
    points_1[0] = np.array([881, 1191])
    bezier_curve_1 = bezier_curve(points_1)

    ##### The second points are for the left right bezier curve

    points_2 = np.array([[250 - 5, 545],
                         [350 - 5, 540],
                         [400 - 5, 465],
                         [435 - 5, 425]], dtype = np.int32)
    points_2 = 4 * points_2
    points_2[3] = np.array([1721, 1700])
    bezier_curve_2 = bezier_curve(points_2)

    ##### The third points are for the up down bezier curve

    points_3 = np.array([[160 - 5, 310],
                         [422 - 5, 228],
                         [512 - 5, 570],
                         [544 - 5, 735]], dtype = np.int32)
    points_3 = 4 * points_3
    points_3[3] = np.array([(4 * resolution) / 2 - 1, 4 * 735])
    bezier_curve_3 = bezier_curve(points_3)

    ##### The forth points are for the up up bezier curve

    points_4 = np.array([[160 - 5, 310],
                         [335 - 5, 170],
                         [495 - 5, 430],
                         [544 - 5, 617]], dtype = np.int32)
    points_4 = 4 * points_4
    points_4[3] = np.array([(4 * resolution) / 2 - 1, 4 * 614])
    bezier_curve_4 = bezier_curve(points_4)

    ##### The fifth points are for the right down bezier curve

    points_5 = np.array([[350 - 5, 550],
                         [450 - 5, 600],
                         [515 - 5, 725],
                         [544 - 5, 850]], dtype = np.int32)
    points_5 = 4 * points_5
    points_5[3] = np.array([(4 * resolution) / 2 - 1, 4 * 850])
    bezier_curve_5 = bezier_curve(points_5)

    ##### The sixth points are for the left up bezier curve shadow

    points_6 = np.array([[260 - 5, 320],
                         [335 - 5, 330],
                         [365 - 5, 385],
                         [410 - 5, 450]], dtype = np.int32)
    points_6 = 4 * points_6
    points_6[0] = np.array([1022, 1276])
    points_6[3] = np.array([1628, 1808])
    bezier_curve_6 = bezier_curve(points_6) 

    ##### The seventh points are for the right up bezier curve shadow

    points_7 = np.array([[425 - 5, 480],
                         [495 - 5, 635],
                         [500 - 5, 725]], dtype = np.int32)
    points_7 = 4 * points_7
    points_7[0] = np.array([1670, 1910])
    points_7[2] = np.array([1986, 2897])
    bezier_curve_7 = bezier_curve(points_7)

    ##### Drawing the fang

    fang = np.array([[480 - 5, 340],
                     [510 - 5, 340],
                     [510 - 5, 400]], dtype = np.int32)
    cv.fillPoly(image, [4 * fang], color = color_fang)

    ##### Drawing the red left shape

    shape_1 = np.concatenate((np.flip(bezier_curve_1, axis = 0), 
                              bezier_curve_3[89:491], 
                              np.flip(bezier_curve_2, axis = 0)))
    cv.fillPoly(image, [shape_1], color = color_red)

    ##### Drawing the up shape

    shape_2 = np.concatenate((bezier_curve_4, 
                              np.flip(bezier_curve_3, axis = 0)))
    cv.fillPoly(image, [shape_2], color = color_fang)

    ##### Drawing the right shape

    shape_3 = np.concatenate((np.array([[4 * (350 - 5), 4 * 550], [4 * (450 - 5), 4 * 450]]), 
                              bezier_curve_3[535:], 
                              np.flip(bezier_curve_5, axis = 0)))
    cv.fillPoly(image, [shape_3], color = color_red)

    ##### Drawing the left shape shadow

    shape_4 = np.concatenate((np.flip(bezier_curve_1[:211], axis = 0), 
                              bezier_curve_3[89:491], 
                              np.flip(bezier_curve_2[801:], axis = 0), 
                              np.flip(bezier_curve_6, axis = 0)))
    cv.fillPoly(image, [shape_4], color = color_red_shadow)

    ##### Drawing the right shape shadow

    shape_5 = np.concatenate((np.array([[1670, 1910]]), 
                              bezier_curve_3[535:], 
                              np.flip(bezier_curve_5[657:], axis = 0), 
                              np.flip(bezier_curve_7[:-1], axis = 0)))
    cv.fillPoly(image, [shape_5], color = color_red_shadow)

    ##### Add contour if requested

    if contour:
        for bezier_curve_i in [bezier_curve_1, bezier_curve_2, bezier_curve_3, bezier_curve_4, bezier_curve_5, bezier_curve_6, bezier_curve_7]:
            cv.polylines(image, [bezier_curve_i], False, color_contour, contour_size)
        cv.line(image, (4 * (350 - 5), 4 * 550), (4 * (450 - 5), 4 * 450), color_contour, contour_size)
        cv.polylines(image, [4 * fang], True, color_contour, contour_size)

    ##### Mirror the image to have a perfect symmetry

    image[:, 2 * resolution:] = cv.flip(image, 1)[:, 2 * resolution:]

    ##### Resize the image to the desired resolution to apply anti-aliasing

    image = cv.resize(image, (resolution, resolution), interpolation = cv.INTER_AREA)
    return image

while True:

    ##### We store the original images to save them on the disk at the full resolution

    images = []
    index = 0

    ##### We loop on every variant possible

    for color_blank in [0, 255]:
        for color_figures in [False, True]:
            if color_figures == False:
                if color_blank == 0:
                    images.append(draw_logo(color_blank, tuple([255 - cc for cc in color_contour]), (0,) * 3, (0,) * 3, (0,) * 3, True, contour_size))

                    ##### We create and move the windows in a 2x3 grid

                    cv.namedWindow(f'Image {index}')
                    cv.moveWindow(f'Image {index}', x_offset + 400 * (index % 3), y_offset - offset + (400 + offset) * (index // 3))
                    cv.imshow(f'Image {index}', cv.resize(images[index], (400, 400), interpolation = cv.INTER_AREA))
                    index += 1
                else:
                    images.append(draw_logo(color_blank, color_contour, (255,) * 3, (255,) * 3, (255,) * 3, 1, contour_size))
                    cv.namedWindow(f'Image {index}')
                    cv.moveWindow(f'Image {index}', x_offset + 400 * (index % 3), y_offset - offset + (400 + offset) * (index // 3))
                    cv.imshow(f'Image {index}', cv.resize(images[index], (400, 400), interpolation = cv.INTER_AREA))
                    index += 1                   
            else:
                for contour in [False, True]:
                    if color_blank == 0:
                        color_fang = (255,) * 3
                    else:
                        color_fang = (0,) * 3
                    images.append(draw_logo(color_blank, color_contour, color_fang, color_red, color_red_shadow, contour, contour_size))
                    cv.namedWindow(f'Image {index}')
                    cv.moveWindow(f'Image {index}', x_offset + 400 * (index % 3), y_offset - offset + (400 + offset) * (index // 3))
                    cv.imshow(f'Image {index}', cv.resize(images[index], (400, 400), interpolation = cv.INTER_AREA))
                    index += 1
    key = cv.waitKey(0) & 0xFF
    if key == 27:
        break

    ##### If we press 's', we save every images to the results folder

    if key == ord('s'):
        for index, image in enumerate(images):
            cv.imwrite(os.path.join('results', f'Image {index}.jpg'), image)
        break

cv.destroyAllWindows()