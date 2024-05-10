from PIL import Image
import numpy as np
import pandas as pd


def calculate_energies(image, rows, cols):
    '''
    Input: image as 2D numpy array, # of rows as int, # of columns as int
    Output: calculated energies as a 2D numpy array

    Function takes the image and returns an 2D array of the calculated energies for each pixel.
    '''
    energies = np.array(([[0] * cols] * rows), dtype="float64")

    for row in range(rows):
        for col in range(cols):
            if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
                energies[row, col] = 1000
            else:
                ver_cost = np.sum(np.power(np.subtract(image[row - 1][col], image[row + 1][col]),2))
                hor_cost = np.sum(np.power(np.subtract(image[row][col - 1], image[row][col + 1]),2))
                energies[row, col] = np.sqrt(ver_cost + hor_cost)
                
    return energies

def find_min_seam(energies):
    '''
    Input: Calculated energies as a 2D array
    Output: An array of the column indices of the min seam
    '''
    rows = len(energies)
    cols = len(energies[0])
    
    # Table meant to be filled with the calculated minimum energies as a 2D Array
    min_energies = energies.copy()
    # Table meant to keep track of the minimum energy path by storing the previous pixel in the mininum path
    traces = energies.copy().astype("int64")


    for row in range(rows):
        for col in range(cols):
            if row == 0: # Case 1: Top row of pixels (smallest subproblem)
                traces[row, col] = 0
            elif col == 0: # Case 2: Have two pixels to check (pixel directly above and diagnal to the right)
                index = np.argmin(min_energies[row - 1, col:col + 2])
                min_energies[row, col] += min_energies[row - 1, col + index]
                traces[row, col] = col + index
            else: # Case 3: Has three or two pixels to check
                index = np.argmin(min_energies[row - 1, col - 1:col + 2])
                min_energies[row, col] += min_energies[row - 1, col - 1 + index]
                traces[row, col] = col - 1 + index
    
    min_seam = [0] * rows
    # Finding the minimum energy in the last row of the table (solution to the largest subproblem)
    min_index = np.argmin(min_energies[rows - 1])

    # Tracing the row above the current minimum energy to build the minumum energy path
    for row in range(rows - 1, 0, -1):
        min_seam[row] = min_index
        min_index = traces[row, min_index]
    
    min_seam[0] = min_index
    return min_seam

def carve_min_seam(image, rows, cols, min_seam):
    '''
    Input: image as 2D numpy array, # of rows as int, # of columns as int, and minimum seam as a array of column indices
    Output: new image with seam carved out as a 2D numpy array

    Citation: We got the idea of using mask and image reshaping from an online source.
    Original Author is Karthik Karanth, Link: https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
    '''
    # Create a mask the same size of the image with values stored as True
    mask = np.array([[True] * cols] * rows)
    
    # Using the minimum seam array of column indices create a mask by setting the pixels to false
    for row in range(rows):
        index = min_seam[row]
        # Set the specific pixel to be removed to False
        mask[row, index] = False
    # Convert the mask into a 3D array 
    mask = np.stack([mask] * 3, axis = 2)
    # Use the mask and reshape to delete minimum seam from image to create a new image
    new_image = image[mask].reshape((rows, cols - 1, 3))

    return new_image

def main():
    # open the image using Image object
    im = Image.open("HJoceanSmall.jpg")
    # Convert image into 2d np array 
    image = np.uint64(im)
    rows, cols, _ = image.shape
    # End result for number fo columns
    new_cols = 408

    # Continue carving seams until desired dimensions
    for i in range(cols - new_cols):
        energies = calculate_energies(image, rows, cols)
        min_seam = find_min_seam(energies)
        # Write the first energy and first min seam into csv files.
        if i == 0:
            pd_energies = pd.DataFrame(energies)
            pd_energies.to_csv("energy.csv", header= False)
            pd_seam = pd.DataFrame(min_seam)
            pd_seam.to_csv("seam1.csv")
        
        image = carve_min_seam(image, rows, cols, min_seam)
        rows, cols, _ = image.shape
    
    # Convert the array into an image 
    new_img = Image.fromarray(image.astype("uint8"))
    # Save the image into a jpg file
    new_img.save("FinalImage.jpg")

if __name__ == '__main__':
    main()