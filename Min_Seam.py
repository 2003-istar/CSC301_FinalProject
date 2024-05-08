from imageio.v2 import imread, imwrite
import numpy as np


def min_seam(energy_arr):
    rows, cols = len(energy_arr), len(energy_arr[0])
    # Create a 2D array to store the energy and arrows to trace back through
    table = [[(0,(0,0)) for _ in range(cols)] for _ in range(rows)]
    
    for row in range(rows):
        for col in range(cols):
            if row == 0: # first row 
                table[row][col] = (energy_arr[row][col], None)
            else: # Checks if the row above exists
                if col == 0: 
                    top = table[row-1][col]
                    top_right = table[row-1][col+1]
                    if top < top_right:
                        table[row][col] = (energy_arr[row][col] + top, (row-1, col))
                    else:
                        table[row][col] = (energy_arr[row][col] + top_right, (row-1, col+1))
                
                if col == cols - 1:
                    ### Pyait got this! Create a function called find_min(paires of indices, 2D array of energies), returns 
                    ### a tuple of minimum energy and a the x,y
                    ### min_energy, (x,y) = find_min()
                
    # Find the min of the last row
    last_row = table[rows-1]
    min_energy = last_row[0]
    min_colInd = None 
    for i in range(cols):
        cur_energy = last_row[i]
        if(cur_energy <= min_energy):
            min_energy = cur_energy
            min_colInd = i
    
    #Trace back through using the locations and store both x and y
    min_seamsList = [0] * rows
    row = rows - 1
    col = min_colInd
    for i in range(rows):
        min_seamsList[rows-1-i] = (row,col)
        row,col = table[row][col][1]

def main():
    image = imread("HJoceanSmall.jpeg")
    print("hello")



if __name__ == '__main__':
    main()              
