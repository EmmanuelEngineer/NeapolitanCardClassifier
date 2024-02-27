import cv2
import numpy as np
import matplotlib as plt

class Config:
    working_directory = "workingDirectory/"
    label_regex = "(.)(.)-(\d+)\.jpg"

def reverse_tuple(t):
    new_tuple = ()
    for i in range(len(t)-1, -1, -1):
        new_tuple += (t[i],)
    return new_tuple


def print_image(*images,titles= None,columns = None):
    # create figure 
    fig = plt.figure(figsize=(10, 8)) 
    
    # setting values to rows and column variables 
    if columns == None:
         columns = len(images)
         rows = 1
    else: 
        a, b = divmod(len(images),columns)
        if b>0:
              rows = a
        else: rows = a+1
    
    for image in images:
        if titles == None:
             title = ""
        # Adds a subplot at the 1st position 
        fig.add_subplot(rows, columns, 1) 
        
        # showing image 
        plt.imshow(image) 
        plt.axis('off') 
        plt.title(title)

def gray_image(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def image_inversion(image):
    return (255-image)

def preprocess_image(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    median_image = cv2.medianBlur(gray_image,3)
    img_contrasty_post_median = cv2.convertScaleAbs(median_image, 1.9, 1)
    th, binary_image = cv2.threshold(img_contrasty_post_median, 155, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image_inversion(binary_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ordered_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return binary_image, tuple(ordered_contours)

def label_properties(path):
    import re
    list_of_matches = re.findall(Config.label_regex, path)
    arr = list_of_matches.pop()
    return {"value": arr[0],"seed": arr[1], "index": arr[2]}

def label_class_only(path):
    import re
    list_of_matches = re.search("(.[A-Z])\.jpg", path)
    
    return {"class": list_of_matches.group(1)}

def images_paths(dataset_directory):
    import glob
    paths =[]
    for x in glob.iglob(dataset_directory + "*.jpg"):
        paths.append(x)
    return paths
