# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 23:18:51 2017

@author: Jorge Morales
"""

import numpy   as np 

def binarize (image, threshold="threshold"):
    """ Binarizes an image passed as a numpy matrix

    Args:
        param1 (np.matrix): The image to binarize.
        param2 (float): The threshold for the binarization. If 0 it will be set to the 90% highest value in the matrix

    Returns:
        np.matrix: The image with values of 0 and 255.

    """
    if threshold=="threshold":
        minValue=image.min()
        maxValue=image.max()        
        threshold=minValue+(maxValue-minValue)*0.9
        
    zeros_indexes= image < threshold
    mask = np.ones(image.shape,dtype=bool) #np.ones_like(a,dtype=bool)
    mask[zeros_indexes] = False
    image[~mask] = 255
    image[mask] = 0
    
    return image

def take_lines_from(url, number=5):
    """ Takes the N first lines of a file and puts them into a new file

    Args:
        url (str): file name and path
        number (int): Number of lines to copy

    Returns:
        Nothing

    """
    try:
        lines=[]
        f=open(url)
        i=0
        for line in f:
            if i>=number:
                break
            lines.append(line)
            i=i+1
        f=open('small_'+url,'w')
        for line in lines:
            f.write(line)
    except Exception as e:
        print(str(e))
        return None

