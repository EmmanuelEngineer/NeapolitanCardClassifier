import cv2
import numpy as np
import matplotlib.pyplot as plt

class Config:
    working_directory = "workingDirectory/"
    class CardTypes:
        with_special_symbols = ["4S","5S","4O","7S"]
        basic_contours = ["AO","AC","AB","AS","2S","3S","3B","2B","2C"]
        contours_to_count=["2O","3O","5O","6O","7O","3C","4C","5C","6C","7C","4B","5B","6B","7B",
        "6S"]#7S
        contours_to_evaluate_color=["9O","9C","9B","9S","8O","8C","8B","8S","RO","RC","RB","RS"]

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
    image = white_balance(image)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    median_image = cv2.medianBlur(gray_image,3)
    img_contrasty_post_median = cv2.convertScaleAbs(median_image, 1.9, 1)
    th, binary_image = cv2.threshold(img_contrasty_post_median, 155, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image_inversion(binary_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ordered_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return binary_image, tuple(ordered_contours)

def average_hue_of_contours(image,contours):
    mask = np.zeros((image.shape[0],image.shape[1],), np.uint8)
    cv2.drawContours(mask, contours, -1, (255), -1)
    HUE_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the average hue within the masked region
    total_hue = 0
    pixel_count = 0

    for contour in contours:
        # Calculate the average hue for each contour
        area = cv2.contourArea(contour)
        if area > 0:
            mask_roi = np.zeros_like(mask)
            cv2.drawContours(mask_roi, [contour], 0, 255, -1)
            hue_values = HUE_image[..., 0][mask_roi == 255]
            total_hue += np.sum(hue_values)
            pixel_count += len(hue_values)
    # Calculate the average hue
    if pixel_count == 0:
        raise("Pixel count in the mask is 0")
    average_hue = total_hue / pixel_count
    return average_hue

def label_properties(path):
    import re
    list_of_matches = re.findall("(.)(.)-(.+)-(\d).jpg", path)
    arr = list_of_matches.pop()
    if arr[0]+arr[1] in (Config.CardTypes.basic_contours + 
                            Config.CardTypes.with_special_symbols):
        label = arr[0]+arr[1]
    elif  arr[0]+arr[1] in Config.CardTypes.contours_to_count:
        label = arr[1]
    else: label = arr[0]
    return {"value": arr[0],"seed": arr[1],"label": label, "index": arr[2]}

def label_properties_generated(path):
    import re
    list_of_matches = re.findall("(.)(.)-.+-(\d).jpg", path)
    arr = list_of_matches.pop()
    if arr[0]+arr[1] in (Config.CardTypes.basic_contours + 
                            Config.CardTypes.with_special_symbols):
        label = arr[0]+arr[1]
    elif  arr[0]+arr[1] in Config.CardTypes.contours_to_count:
        label = arr[1]
    else: label = arr[0]
    return {"value": arr[0],"seed": arr[1],"label": label, "index": arr[2]}

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def label_class_only(path):
    import re
    list_of_matches = re.findall("(.)([A-Z])\.jpg", path)
    arr = list_of_matches.pop()
    if arr[0]+arr[1] in Config.CardTypes.basic_contours:
        label = arr[0]+arr[1]
    elif  arr[0]+arr[1] in Config.CardTypes.contours_to_count:
        label = arr[1]
    else: label = arr[0]
    return {"value": arr[0],"seed": arr[1],"label": label}

def images_paths(dataset_directory):
    import glob
    paths =[]
    for x in glob.iglob(dataset_directory + "*.jpg"):
        paths.append(x)
    return paths

def print_used(image,contours):
    cv2.drawContours(image, contours[0], -1, (0, 255, 0), 2)  
    print(image)