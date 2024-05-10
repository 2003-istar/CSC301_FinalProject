from imageio.v2 import imread, imwrite
from PIL import Image
import math
import numpy as np
import pandas as pd

def calculate_energies(image, rows, cols):
    '''
    '''
    energies = np.array(([[0] * cols] * rows), dtype="float64")
    print(rows)
    print(cols)
    for row in range(rows):
        for col in range(cols):
            if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
                energies[row, col] = 1000
            else:
                ver_cost = np.sum(np.power(np.subtract(image[row - 1][col], image[row + 1][col]),2))
                hor_cost = np.sum(np.power(np.subtract(image[row][col - 1], image[row][col + 1]),2))
                if row == 1 and col == 2:
                    print(np.subtract(image[row - 1][col], image[row + 1][col]))
                    print(np.power(np.subtract(image[row - 1][col], image[row + 1][col]),2))
                    print(ver_cost)
                    print(hor_cost)
                energies[row, col] = np.sqrt(ver_cost + hor_cost)
                
    print('CE')
    return energies

def find_min_seam(energies):
    '''
    '''
    rows = len(energies)
    cols = len(energies[0])
    min_energies = energies.copy()
    traces = energies.copy().astype("int64")

    for row in range(rows):
        for col in range(cols):
            if row == 0:
                traces[row, col] = 0
            elif col == 0:
                index = np.argmin(min_energies[row - 1, col:col + 2])
                min_energies[row, col] += min_energies[row - 1, col + index]
                traces[row, col] = col + index
            else: 
                index = np.argmin(min_energies[row - 1, col - 1:col + 2])
                min_energies[row, col] += min_energies[row - 1, col - 1 + index]
                traces[row, col] = col - 1 + index
    
    min_seam = [0] * rows
    min_index = np.argmin(min_energies[rows - 1])

    for row in range(rows - 1, 0, -1):
        min_seam[row] = min_index
        min_index = traces[row, min_index]
    
    min_seam[0] = min_index
    print('FMS')
    return min_seam

def carve_min_seam(image, rows, cols, min_seam):
    '''
    This function ...

    Args:
        image: 
        seam: 
    Returns:
        image: 
    '''
    mask = np.array([[True] * cols] * rows)
    
    for row in range(rows):
        index = min_seam[row]
        mask[row, index] = False
    
    mask = np.stack([mask] * 3, axis = 2)

    new_image = image[mask].reshape((rows, cols - 1, 3))

    print('CMS')
    return new_image

def main():
    # image = imread('HJoceanSmall.jpg')
    im = Image.open("HJoceanSmall.jpg")
    image = np.uint64(im)
    rows, cols, _ = image.shape
    new_cols = 408
    energies = calculate_energies(image, rows, cols)
    min_seam = find_min_seam(energies)


    pd_energies = pd.DataFrame(energies)
    pd_energies.to_csv("energy.csv", header= False)
    pd_seam = pd.DataFrame(min_seam)
    pd_seam.to_csv("seam1.csv")
    for i in range(cols - new_cols):
        energies = calculate_energies(image, rows, cols)
        min_seam = find_min_seam(energies)
        image = carve_min_seam(image, rows, cols, min_seam)
        rows, cols, _ = image.shape
    
    
    # imwrite('New_HJoceanSmall.jpg', image)
    new_img = Image.fromarray(image.astype("uint8"))
    new_img.save("FinalImage.jpg")

if __name__ == '__main__':
    main()