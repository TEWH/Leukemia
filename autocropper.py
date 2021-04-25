#importing libraries
import collections, functools, operator
import cv2 as cv
from PIL import Image
import numpy as np
import json
import os
import time

def timeit(func):
    def wrapper(*args, **kwargs):
        now = time.time()
        retval = func(*args, **kwargs)
        print('{} took {:.5f}s'.format(func.__name__, time.time() - now))
        return retval
    return wrapper

#given a filename and vgg formattted json file path from makesense.ai or other image annotation software,
#this function iterates through the json file and returns the x and y coordinates as lists
def coordinates_from_json(filename, json_path):
    file = open(json_path)
    data = json.load(file)
    file.close() #we now have the json object as a python dictionary
    x_list = data[filename]['regions']['0']['shape_attributes']['all_points_x']
    y_list = data[filename]['regions']['0']['shape_attributes']['all_points_y']
    return x_list, y_list


#the following function takes a filename of an image, a json object containing coordinate information,
#and a directory (str) in which to store the resulting output, the mask
def mask_from_file(image_dir, filename, json_path, mask_dir, n=1):
    x_list, y_list = coordinates_from_json(filename, json_path)
    path = os.path.abspath(image_dir + '/' + filename)
    image = Image.open(path)
    shape = image.size
    #print(shape)
    image.close()
    contours = np.stack((x_list, y_list), axis = 1)
    polygon = np.array([contours], dtype = np.int32)
    zero_mask = np.zeros((shape[1], shape[0]), np.uint8)
    polyMask = cv.fillPoly(zero_mask, polygon, n)
    cv.imwrite(mask_dir + '/' + filename[:-4] + '_mask.png', polyMask)
    return polyMask

# This finds the x and y coordinates of the start and end positions, used to find the centroid of the image
# Returns a dictionary containing the start position, and the end position
# To be used with simpleCentroid function

# @timeit
def numpy_centroid(matrix):
	
	rows = [i for i in range(len(matrix))] # get a list of indexes for all rows of the matrix
	
	columns = [j for j in range(len(matrix[0]))] # get a list of indexes for all columns of the matrix
	
	# convert the lists to numpy arrays
	r = np.array(rows)
	c = np.array(columns)
	
	def value_max_width_len(values):
	
		j = values[np.fromiter(map(len, values), int).argmax()] # use this to get the longest array from an array of arrays
		#v = max(map(len, values))
		return j

	def numpy2centroid(matrix, index_array):
		
		mat = index_array * matrix # broadcast the array of indexes to all the arrays of the matrix, multiply together
		# will get matrix of 0s and the indexes
		
		
		mat = [n[n != 0] for n in mat] # remove all zeros, so now we have a matrix of arrays with either only indexes, or empty arrays
		# convert the list back into an array
		matrix = np.array(mat)
		
		# x is now the longest array of the matrix (remember, without zeros, the lengths of the index arrays will differ based on the length of the polygon)
		x = value_max_width_len(matrix)
		j = x[-1] - x[0] # subtract the last index from the first index
		
		center = x[0] + j // 2 # divide the difference between the indexes in half, and add that to first index to find the center
		
		return {"center":center, "max_len":j}
		
	x = numpy2centroid(matrix, c) # this will give you the x coordinate of the centroid, from our calculations using the row indexes
	x_coord = x["center"]
	max_x = x["max_len"]
	trans_matrix = np.transpose(matrix) # transpose the matrix, so that the columns become the rows, and the rows become the columns
	y = numpy2centroid(trans_matrix, r) # use the transposed matrix and the column index array to find the y coordinates of the centroid
	y_coord = y["center"]
	max_y = y["max_len"]
	
	
	return {'x': x_coord, 'y': y_coord, 'max_x':max_x, 'max_y':max_y} # returns a dictionary with the x and y coordinates
		

#given the filepath of an image (in .jpg) and its corresponding mask of ones and zeros,
#this function crops and stores the image in the same directory
#where n is half the length of the desired width and height
# @timeit
def cropper(image_dir, filename, matrix, out_dir = "./", n = 1000, j = 200, k = 400, extension = ".jpg"):

    path = os.path.abspath(image_dir + '/' + filename)
    img = cv.imread(path)
               
    coords = numpy_centroid(matrix)
    print(coords)
    center_x, center_y = coords['x'], coords['y']
    
    max_size = max([coords['max_x'], coords['max_y']])
    
    # this performs same function as previous n_finder function
    if max_size >= n:
    	n = max_size + j
    elif max_size <= 400:
    	n = max_size + k
    else:
    	pass
    
    left_bound = center_x - n // 2 if (center_x - n // 2) >= 0 else 0
    right_bound = center_x + n // 2 if (center_x + n) < len(matrix[0]) else (len(matrix[0]) - 1)
    bottom_bound = center_y + n // 2 if (center_y + n) < len(matrix) else len(matrix) - 1
    top_bound = center_y - n // 2 if (center_y - n) >= 0 else 0

    cropped_img = img[top_bound:bottom_bound, left_bound:right_bound]
    cropped_path = out_dir + filename[:-4] + "_cropped" + extension
    print(cropped_path)
        
    cv.imwrite(cropped_path, cropped_img)
	
#iterates through files in a folder to create masks (takes files from filepath and json annotations from annotations and places masks in mask_dir)
def mask_from_annotations(filepath, annotations, mask_dir):

    for f in os.listdir(filepath):

        mask_from_file(filepath, f, annotations,mask_dir)

#PA - Petrous Annotations
#TA - Teeth Annotations
#A - Alaukik
#K - Kushal
