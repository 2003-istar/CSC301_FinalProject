from imageio.v2 import imread, imwrite
import math
import numpy as np

def calculate_energies(image, rows, cols):
    '''
    '''
    energies = np.array([[0] * cols] * rows)
    
    for row in range(rows):
        for col in range(cols):
            if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
                energies[row, col] = 1000
            else:
                r1, g1, b1 = image[row, col - 1]
                r2, g2, b2 = image[row, col + 1]
                hor_cost = ((r1 - r2) ** 2) + ((g1 - g2) ** 2) + ((b1 - b2) ** 2)

                r1, g1, b1 = image[row - 1, col]
                r2, g2, b2 = image[row + 1, col]
                ver_cost = ((r1 - r2) ** 2) + ((g1 - g2) ** 2) + ((b1 - b2) ** 2)

                energies[row, col] = math.sqrt(hor_cost + ver_cost)
    print('CE')
    return energies

def find_min_seam(energies):
    '''
    '''
    rows = len(energies)
    cols = len(energies[0])
    min_energies = energies.copy()
    traces = energies.copy()

    for row in range(rows):
        for col in range(cols):
            if row == 0:
                traces[row, col] = -1
            elif col == 0:
                index = np.argmin(energies[row - 1, col:col + 2])
                min_energies[row, col] += min_energies[row - 1, col + index]
                traces[row, col] = col + index
            else:
                index = np.argmin(energies[row - 1, col - 1:col + 2])
                min_energies[row, col] += min_energies[row - 1, col - 1 + index]
                traces[row] = col - 1 + index
    
    min_seam = [0] * rows
    min_index = np.argmin(min_energies[rows - 1])

    for row in range(rows - 1, -1, -1):
        min_seam[row] = min_index
        min_index = traces[row, min_index]
    
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
    image = imread('HJoceanSmall.jpg')
    rows, cols, _ = image.shape
    new_cols = 408

    for i in range(cols - new_cols):
        energies = calculate_energies(image, rows, cols)
        min_seam = find_min_seam(energies)
        image = carve_min_seam(image, rows, cols, min_seam)
        rows, cols, _ = image.shape

    imwrite('New_HJoceanSmall.jpg', image)

if __name__ == '__main__':
    main()