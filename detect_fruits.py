import json
from pathlib import Path
from typing import Dict

import click
import cv2
import numpy as np
from tqdm import tqdm

def detect_orange(img):
    height, width, channels = img.shape
    if width < height:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    scaled_img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    gaussian_img = cv2.GaussianBlur(scaled_img, (9, 9), 0)
    hsv = cv2.cvtColor(gaussian_img, cv2.COLOR_BGR2HSV)
    w = np.array([12, 193, 180])
    s = np.array([18, 249, 235])
    kernel = np.ones((33, 33), np.uint8)
    mask = cv2.inRange(hsv, w, s)
    ker_mask = cv2.dilate(mask, kernel, iterations=1)
    ker_mask = cv2.morphologyEx(ker_mask, cv2.MORPH_CLOSE, kernel)
    ker_mask = cv2.morphologyEx(ker_mask, cv2.MORPH_OPEN, kernel)
    res = cv2.bitwise_and(gaussian_img, gaussian_img, mask=ker_mask)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow('image', thresh)
    # cv2.waitKey()
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours_number = []
    for contour in contours:
        if cv2.contourArea(contour) > 6000:
            contours_number.append(contour)
    return(len(contours_number))


def detect_banana(img):
    height, width, channels = img.shape
    if width < height:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    scaled_img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    gaussian_img = cv2.GaussianBlur(scaled_img, (5, 5), 0)
    hsv = cv2.cvtColor(gaussian_img, cv2.COLOR_BGR2HSV)
    w = np.array([20, 93, 116])
    s = np.array([36, 230, 240])
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.inRange(hsv, w, s)
    ker_mask = cv2.dilate(mask, kernel, iterations=1)
    ker_mask = cv2.morphologyEx(ker_mask, cv2.MORPH_CLOSE, kernel)
    ker_mask = cv2.morphologyEx(ker_mask, cv2.MORPH_OPEN, kernel)
    res = cv2.bitwise_and(gaussian_img, gaussian_img, mask=ker_mask)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow('image', thresh)
    # cv2.waitKey()
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours_number = []
    for contour in contours:
        if cv2.contourArea(contour) > 7000:
            contours_number.append(contour)
    return(len(contours_number))

def detect_apple(img):
    height, width, channels = img.shape
    if width < height:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    scaled_img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    gaussian_img = cv2.GaussianBlur(scaled_img, (5, 5), 0)
    hsv = cv2.cvtColor(gaussian_img, cv2.COLOR_BGR2HSV)

    w1 = np.array([0, 33, 29])
    s1 = np.array([3, 194, 188])
    kernel1 = np.ones((21, 21), np.uint8)
    mask1 = cv2.inRange(hsv, w1, s1)
    ker_mask1 = cv2.dilate(mask1, kernel1, iterations=1)
    ker_mask1 = cv2.morphologyEx(ker_mask1, cv2.MORPH_CLOSE, kernel1)
    ker_mask1 = cv2.morphologyEx(ker_mask1, cv2.MORPH_OPEN, kernel1)
    w2 = np.array([3, 100, 44])
    s2 = np.array([17, 219, 160])
    kernel2 = np.ones((13, 13), np.uint8)
    mask2 = cv2.inRange(hsv, w2, s2)
    ker_mask2 = cv2.dilate(mask2, kernel2, iterations=1)
    ker_mask2 = cv2.morphologyEx(ker_mask2, cv2.MORPH_CLOSE, kernel2)
    ker_mask2 = cv2.morphologyEx(ker_mask2, cv2.MORPH_OPEN, kernel2)
    ker_mask = ker_mask1 + ker_mask2
    res = cv2.bitwise_and(gaussian_img, gaussian_img, mask=ker_mask)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow('image', thresh)
    # cv2.waitKey()
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours_number = []
    for contour in contours:
        if cv2.contourArea(contour) > 6000:
            contours_number.append(contour)
    return(len(contours_number))


def detect_fruits(img_path: str) -> Dict[str, int]:
    """Fruit detection function, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each fruit.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    #TODO: Implement detection method.
    
    apple = 0
    banana = 0
    orange = 0

    apple = detect_apple(img)
    banana = detect_banana(img)
    orange = detect_orange(img)

    return {'apple': apple, 'banana': banana, 'orange': orange}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False,
                                                                                  path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}
    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
