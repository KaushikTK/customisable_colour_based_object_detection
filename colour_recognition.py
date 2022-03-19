import cv2
import numpy as np

def load_image(path:str):
    img = cv2.imread(path)
    return img

def define_color_boundary(colour:str):
    # [lower_BGR, upper_BGR] is the format
    red = [[17, 15, 100], [50, 56, 200]]
    if colour == 'red': return [np.array(red[0], dtype="uint8"), np.array(red[1], dtype="uint8")]

def find_pixels_in_range(img, low, up):
    mask = cv2.inRange(img, low, up)
    output = cv2.bitwise_and(img, img, mask=mask)
    return mask, output

def create_image_negative(img):
    return cv2.bitwise_not(img)

def find_countours(img):
    thresh = cv2.threshold(img, 255, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, heirarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # extract only the external contours and remove contours with 0 area size
    external_contours = []
    length = len(contours)
    for i in range(length):
        if(heirarchy[0,i,3] == -1): external_contours.append(contours[i])
    contours = external_contours.copy()
    external_contours = []
    length = len(contours)
    for i in contours:
        if cv2.contourArea(i) > 0: external_contours.append(i)
    return external_contours

def draw_bounding_box(img, contours):
    boxes = []
    for i in contours:
        (x, y, w, h) = cv2.boundingRect(i)
        boxes.append([x,y, x+w,y+h])
    boxes = np.asarray(boxes)
    left, top = np.min(boxes, axis=0)[:2]
    right, bottom = np.max(boxes, axis=0)[2:]
    img = cv2.rectangle(img,(left,top),(right,bottom),(255,0,0),2)
    cv2.imshow('output', img)
    cv2.waitKey(0)


def main():
    #path = '../pictures/apple.PNG'
    path = '../pictures/apple2.jpg'
    colour = 'red' # update the color and boundaries as per requirement
    img = load_image(path)
    lower, upper = define_color_boundary(colour)
    mask, _ = find_pixels_in_range(img, lower, upper)
    img_negative = create_image_negative(mask)
    contours = find_countours(img_negative)
    draw_bounding_box(img, contours)


main()