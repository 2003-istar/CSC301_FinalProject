# Import Image from PIL package
from PIL import Image
import math


def energy_calc(im):
    # Create 2D array to store energy of image
    rows, cols = im.size 
    arr = [[0] * cols] *rows
    
    for x in range(0, rows):
        for y in range(0, cols):
            if(x == 0 or x == (rows - 1) or y == 0 or y == (cols - 1)): #If edge, energy = 1000
                arr[x][y] = 1000
            else:
                # Calculate Vertical cost
                r1, g1, b1 = im.getpixel((x-1, y)) #pixel on top of cur
                r2, g2, b2 = im.getpixel((x+1, y)) #pixel below cur
                vertical = ((r1 - r2) ** 2 ) + ((g1 - g2) ** 2 ) + ((b1 - b2) ** 2 )
                
                # Calculate Horizontal cost
                r1, g1, b1 = im.getpixel((x, y-1)) #pixel to the left of cur
                r2, g2, b2 = im.getpixel((x, y+1)) #pixel to the right of cur
                horizontal = ((r1 - r2) ** 2 ) + ((g1 - g2) ** 2 ) + ((b1 - b2) ** 2 )
                
                # Combine and take square root
                arr[x][y] = math.sqrt(vertical + horizontal) 
    return arr



# Create an image object of original image that gets passed into the function
img = Image.open(r"/workspaces/Energy-Calculator/inputimage.jpg")
energy_calc(img)


