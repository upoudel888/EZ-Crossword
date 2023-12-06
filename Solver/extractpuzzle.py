import cv2
import numpy as np
import math
from sklearn.linear_model import LinearRegression
import pytesseract
import re
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/hello/Tesseract-OCR/tesseract.exe'
image_path = "try heree.jpg"

def first_preprocessing(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray,75,25)
    contours,hierarchies = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours,key = cv2.contourArea,reverse = True)
    largest_contour = sorted_contours[0]
    box = cv2.boundingRect(sorted_contours[0])
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    result = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return result

def remove_head(image):
    custom_config = r'--oem 3 --psm 6'  # Tesseract OCR configuration
    detected_text = pytesseract.image_to_string(image, config=custom_config)
    lines = detected_text.split('\n')

# Find the first line containing some text
    line_index = 0
    for i, line in enumerate(lines):
        if line.strip() != '':
            line_index = i
            break
    first_newline_idx = detected_text.find('\n')
    result = cv2.rectangle(image, (0, line_index), (image.shape[1], first_newline_idx), (255,255,255), thickness=cv2.FILLED)
    return result

def second_preprocessing(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray,75,25)
    contours,hierarchies = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours,key = cv2.contourArea,reverse = True)
    largest_contour = sorted_contours[0]
    box2 = cv2.boundingRect(sorted_contours[0])
    x = box2[0]
    y = box2[1]
    w = box2[2]
    h = box2[3]
    result2 = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return result2

def find_vertical_profile(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    vertical_profile = np.sum(binary, axis=0)
    return vertical_profile

def detect_steepest_changes(projection_profile, threshold=0.4, start_idx=0, min_valley_width=10, min_search_width=50):
    differences = np.diff(projection_profile)
    change_points = np.where(np.abs(differences) > threshold * np.max(np.abs(differences)))[0]
    left_boundaries = []
    right_boundaries = []

    for idx in change_points:
        if idx <= start_idx:
            continue

        if idx - start_idx >= min_search_width:
            decreasing_profile = projection_profile[idx:]
            if np.any(decreasing_profile > 0):
                right_boundary = idx + np.argmin(decreasing_profile)
                right_boundaries.append(right_boundary)
            else:
                continue
            valley_start = max(start_idx, idx - min_valley_width)
            valley_start = valley_start-40
            valley_end = min(idx + min_valley_width, len(projection_profile) - 1)
            valley = valley_start + np.argmin(projection_profile[valley_start:valley_end])
            left_boundaries.append(valley)

            break

    return left_boundaries, right_boundaries

def crop_text_columns(image, projection_profile, threshold=0.4):
    start_idx = 0
    text_columns = []

    while True:
        left_boundaries, right_boundaries = detect_steepest_changes(projection_profile, threshold, start_idx)
        if not left_boundaries or not right_boundaries:
            break
        left = left_boundaries[0]
        right = right_boundaries[0]
        text_column = image[:, left:right]
        text_columns.append(text_column)

        start_idx = right

    return text_columns


def parse_clues(clue_text):
    lines = clue_text.split('\n')
    clues = {}
    number = None
    column = 0
    for line in lines:
        if "column separation" in line:
            column += 1
            continue
        pattern = r"^(\d+(?:\.\d+)?)\s*(.+)"  # Updated pattern to handle decimal point numbers for clues
        match = re.search(pattern, line)
        if match:
            number = float(match.group(1))  # Convert the matched number to float if there is a decimal point
            if number not in clues:
                clues[number] = [column,match.group(2).strip()]
            else:
                continue
        elif number is None:
            continue
        elif clues[number][0] != column:
            continue
        else:
            clues[number][1] += " " + line.strip()  # Append to the previous clue if it's a multiline clue

    return clues

def parse_crossword_clues(text):
    # Check if "Down" clues are present
    match = re.search(r'[dD][oO][wW][nN]\n', text)
    if match:
        across_clues, down_clues = re.split(r'[dD][oO][wW][nN]\n', text)
    else:
        # If "Down" clues are not present, set down_clues to an empty string
        across_clues, down_clues = text, ""

    across = parse_clues(across_clues)
    down = parse_clues(down_clues)

    return across, down


def classify_text(filtered_columns):
    text = ""
    custom_config = r'--oem 3 --psm 6'
    for i, column in enumerate(filtered_columns):
        column2 = cv2.cvtColor(column, cv2.COLOR_BGR2RGB)
        scale_factor = 2.0  # You can adjust this value

# Calculate the new dimensions after scaling
        new_width = int(column2.shape[1] * scale_factor)
        new_height = int(column2.shape[0] * scale_factor)

# Resize the image using OpenCV
        scaled_image = cv2.resize(column2, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# Apply image enhancement techniques
        denoised_image = cv2.fastNlMeansDenoising(scaled_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        enhanced_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale  # Apply histogram equalization
        detected_text = pytesseract.image_to_string(enhanced_image, config=custom_config)
    # print(detected_text)
        text+=detected_text
    across_clues, down_clues = parse_crossword_clues(text)
    return across_clues,down_clues

def get_text(image):
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    result = first_preprocessing(image)
    result1 = remove_head(result)
    result2 = second_preprocessing(result1)
    vertical_profile = find_vertical_profile(result2)
    combined_columns = crop_text_columns(result2,vertical_profile)
    across,down = classify_text(combined_columns)
    return across,down


################################ Grid Extraction begins here ###########################
########################################################################################


# for applying non max suppression of the contours
def calculate_iou(image, contour1, contour2):
    # Create masks for each contour
    mask1 = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(mask1, [contour1], -1, 255, thickness=cv2.FILLED)
    
    mask2 = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(mask2, [contour2], -1, 255, thickness=cv2.FILLED)

    # Find the intersection between the two masks
    intersection = cv2.bitwise_and(mask1, mask2)

    # Calculate the intersection area
    intersection_area = cv2.countNonZero(intersection)

    # Calculate the union area (Not the accurate one but works alright XD !)
    union_area = cv2.contourArea(cv2.convexHull(np.concatenate((contour1, contour2))))

    # Calculate the IoU
    iou = intersection_area / union_area
    return iou

# remove overlapping contours, non square and not quardatic contours
# this check every contour with every other contour so be careful
def filter_contours(img_gray2, contours, iou_threshold = 0.6, asp_ratio = 1,tolerance = 0.5):
    # Remove overlapping contours, removing that are not square
    filtered_contours = []
    epsilon = 0.02
    for contour in contours:
        
        # Approximate the contour to reduce the number of points
        epsilon_multiplier = epsilon * cv2.arcLength(contour, True)
        approximated_contour = cv2.approxPolyDP(contour, epsilon_multiplier, True)
        
        # find the aspect ratio of the contour, if it is close to 1 then keep it otherwise discard
        _,_,w,h = cv2.boundingRect(approximated_contour)
        if(abs(float(w)/h - asp_ratio) > tolerance ): continue
        
        # Calculate the IoU with all existing contours
        iou_values = [calculate_iou(img_gray2,np.array(approximated_contour), np.array(existing_contour)) for existing_contour in filtered_contours]

        # If the IoU value with all existing contours is below the threshold, add the current contour
        if not any(iou_value > iou_threshold for iou_value in iou_values):
            filtered_contours.append(approximated_contour)

    return filtered_contours

# https://stackoverflow.com/questions/383480/intersection-of-two-lines-defined-in-rho-theta-parameterization/383527#383527
# Define the parametricIntersect function
def parametricIntersect(r1, t1, r2, t2):
    ct1 = np.cos(t1)
    st1 = np.sin(t1)
    ct2 = np.cos(t2)
    st2 = np.sin(t2)
    d = ct1 * st2 - st1 * ct2
    if d != 0.0:
        x = int((st2 * r1 - st1 * r2) / d)
        y = int((-ct2 * r1 + ct1 * r2) / d)
        return x, y
    else:
        return None

# Group the coordinate to a list such that each point in a list may belong to a line
def group_lines(coordinates,axis=0,threshold=10):
    sorted_coordinates = list(sorted(coordinates,key=lambda x: x[axis]))
    groups = []
    current_group = []

    for i in range(len(sorted_coordinates)):
        if i!=0 and abs(current_group[0][axis] - sorted_coordinates[i][axis]) > threshold: # condition to change the group
            if len(current_group) > 4:
                groups.append(current_group)
                current_group = []
        current_group.append(sorted_coordinates[i]) # condition to append to the group
    if(len(current_group) > 4):
        groups.append(current_group)
    return groups

# Use the Grouped Lines to Fit a line using Linear Regression
def fit_lines(grouped_lines,is_horizontal = False):
    actual_lines = []
    for coordinates in grouped_lines:
        # Converting into numpy array
        coordinates_arr = np.array(coordinates)
        # Separate the x and y coordinates
        x = coordinates_arr[:, 0]
        y = coordinates_arr[:, 1]
        # Fit a linear regression model
        regressor = LinearRegression()
        regressor.fit(y.reshape(-1, 1), x)
        # Get the slope and intercept of the fitted line
        slope = regressor.coef_[0]
        intercept = regressor.intercept_

        if(is_horizontal):
            intercept = np.mean(y)
        actual_lines.append((slope,intercept))
    
    return actual_lines

# Calculates difference between two consecutive elements in an array
def average_distance(arr):
    n = len(arr)
    distance_sum = 0

    for i in range(n - 1):
        distance_sum += abs(arr[i+1] - arr[i])

    average = distance_sum / (n - 1)
    return average

# If two adjacent lines are near than some threshold, then merge them
# Returns Results in y = mx + b from
def average_out_similar_lines(lines_m_c,lines_coord,del_threshold,is_horizontal=False):
    averaged_lines = []
    i = 0
    while(i < len(lines_m_c) - 1):

        _, intercept1 = lines_m_c[i]
        _, intercept2 = lines_m_c[i + 1]

        if abs(intercept2 - intercept1) < del_threshold:
            new_points = np.array(lines_coord[i] + lines_coord[i+1][:-1])
            # Separate the x and y coordinates
            x = new_points[:, 0]
            y = new_points[:, 1]

            # Fit a linear regression model
            regressor = LinearRegression()
            regressor.fit(y.reshape(-1, 1), x)

            # Get the slope and intercept of the fitted line
            slope = regressor.coef_[0]
            intercept = regressor.intercept_

            if(is_horizontal):
                intercept = np.mean(y)
            averaged_lines.append((slope,intercept))
            i+=2
        else:
            averaged_lines.append(lines_m_c[i])
            i+=1
    if(i < len(lines_m_c)):
        averaged_lines.append(lines_m_c[i])

    return averaged_lines

# If two adjacent lines are near than some threshold, then merge them
# Returns Results in normalized vector form
def average_out_similar_lines1(lines_m_c,lines_coord,del_threshold):
    averaged_lines = []
    i = 0
    while(i < len(lines_m_c) - 1):

        _, intercept1 = lines_m_c[i]
        _, intercept2 = lines_m_c[i + 1]

        if abs(intercept2 - intercept1) < del_threshold:
            new_points = np.array(lines_coord[i] + lines_coord[i+1][:-1])
            coordinates = np.array(new_points)
            points = coordinates[:, None, :].astype(np.int32)
            # Fit a line using linear regression
            [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            averaged_lines.append((vx, vy, x, y))
            i+=2
        else:
            new_points = np.array(lines_coord[i])

            coordinates = np.array(new_points)
            points = coordinates[:, None, :].astype(np.int32)
            # Fit a line using linear regression
            [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            averaged_lines.append((vx, vy, x, y))
            i+=1
    if(i < len(lines_m_c)):
        new_points = np.array(lines_coord[i])
        coordinates = np.array(new_points)
        points = coordinates[:, None, :].astype(np.int32)
        # Fit a line using linear regression
        [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        averaged_lines.append((vx, vy, x, y))

    return averaged_lines

def get_square_color(image, box):

    # Determine the size of the square region
    square_size = (box[1][0] - box[0][0]) / 3

    # Determine the coordinates of the square region inside the box
    top_left = (box[0][0] + square_size, box[0][1] + square_size)
    bottom_right = (box[0][0] + square_size*2, box[0][1] + square_size*2)

    # Extract the square region from the image
    square_region = image[int(top_left[1]):int(bottom_right[1]), int(top_left[0]):int(bottom_right[0])]

    # Calculate the mean pixel value of the square region
    mean_value = np.mean(square_region)

    # Determine whether the square region is predominantly black or white
    if mean_value < 128:
        square_color = "."
    else:
        square_color = " "

    return square_color

# accepts image in grayscale
def extract_grid(image):

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Apply dilation to connect nearby edges and make them more contiguous
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # # Applying canny edge detector
    # detecting contours on the canny image
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # sorting the contours by the descending order area of the contour
    sorted_contours = list(sorted(contours, key=cv2.contourArea,reverse=True))
    # filtering out the top 10 largest by applying NMS and only selecting square ones (Apsect ratio 1)
    filtered_contours = filter_contours(image, sorted_contours[0:10],iou_threshold=0.6,asp_ratio=1,tolerance=0.2)
    
    # largest Contour Extraction
    largest_contour = []
    if(len(filtered_contours)):
        largest_contour = filtered_contours[0]
    else:
        largest_contour = sorted_contours[0]

    # --- Performing Perspective warp of the largest contour ---
    coordinates_list = []

    if(largest_contour.shape != (4,1,2)):
        largest_contour = cv2.convexHull(largest_contour)
        if(largest_contour.shape != (4,1,2)):
            rect = cv2.minAreaRect(largest_contour)
            largest_contour = cv2.boxPoints(rect)
            largest_contour = largest_contour.astype('int')

    coordinates_list = largest_contour.reshape(4, 2).tolist()

    # Convert coordinates_list to a numpy array
    coordinates_array = np.array(coordinates_list)

    # Find the convex hull of the points
    hull = cv2.convexHull(coordinates_array)

    # Find the extreme points of the convex hull
    extreme_points = np.squeeze(hull)

    # Sort the extreme points by their x and y coordinates to determine the order
    sorted_points = extreme_points[np.lexsort((extreme_points[:, 1], extreme_points[:, 0]))]

    # Extract top left, bottom right, top right, and bottom left points
    tl = sorted_points[0]
    tr = sorted_points[1]
    bl = sorted_points[2]
    br = sorted_points[3]

    if(tr[1] < tl[1]):
        tl,tr = tr,tl
    if(br[1] < bl[1]):
        bl,br = br,bl

    # Define pts1
    pts1 = [tl, bl, tr, br]

    # Calculate the bounding rectangle coordinates
    x, y, w, h = 0,0,400,400
    # Define pts2 as the corners of the bounding rectangle
    pts2 = [[3, 3], [400, 3], [3, 400], [400, 400]]

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))

    # Apply the perspective transformation to the cropped_image
    transformed_img = cv2.warpPerspective(image, matrix, (403, 403))
    cropped_image = transformed_img.copy()

    plt.figure(figsize=(12,8))
    plt.axis("off")
    plt.imsave("noice1.jpg",cv2.cvtColor(cropped_image,cv2.COLOR_GRAY2RGB))

    # if the largest contour was not exactly quadilateral

    # -- Performing Hough Transform --

    similarity_threshold = math.floor(w/30) # Thresholds for filtering Similar Hough Lines

    # Applying Gaussian Blur to reduce noice and improve dege detection
    blurred = cv2.GaussianBlur(cropped_image, (5, 5), 0)
    # Perform Canny edge detection on the GrayScale Image
    edges = cv2.Canny(blurred, 50, 150) 
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    # Filter out similar lines
    filtered_lines = []
    for line in lines:
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            rho, theta = arr
            is_similar = False
            for filtered_line in filtered_lines:
                filtered_rho, filtered_theta = filtered_line
                # similarity threshold is 10
                if abs(rho - filtered_rho) < similarity_threshold and abs(theta - filtered_theta) < np.pi/180 * similarity_threshold:
                    is_similar = True
                    break
            if not is_similar:
                filtered_lines.append((rho, theta))

    # Filter out the horizontal and the vertical lines
    horizontal_lines = []
    vertical_lines = []
    for rho, theta in filtered_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        slope = (y2 - y1) / (x2 - x1 + 0.0001)
        # do taninv(0.17) it is nearly equal to 10
        if( abs(slope) <= 0.18 ):
            horizontal_lines.append((rho,theta))
        elif (abs(slope) > 6):
            vertical_lines.append((rho,theta))

    # Find the intersection points of horizontal and vertical lines
    hough_corners = []
    for h_rho, h_theta in horizontal_lines:
        for v_rho, v_theta in vertical_lines:
            x, y = parametricIntersect(h_rho, h_theta, v_rho, v_theta)
            if x is not None and y is not None:
                hough_corners.append((x, y))

    # -- Performing Harris Corner Detection --

    # Create CLAHE object with specified clip limit
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    clahe_image = clahe.apply(cropped_image)

    # harris corner detection for CLHAE IMAGE
    dst = cv2.cornerHarris(clahe_image,2,3,0.04)
    ret,dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    dst = cv2.dilate(dst,None)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TermCriteria_MAX_ITER,100,0.001)
    harris_corners = cv2.cornerSubPix(clahe_image,np.float32(centroids),(5,5),(-1,-1),criteria)

    drawn_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
    for i in harris_corners:
        x,y = i
        image2 = cv2.circle(drawn_image, (int(x),int(y)), radius=0, color=(0, 0, 255), thickness=3)

    # -- Using Regression Model to approximate horizontal and vertical Lines

    # reducing to 0 decimal places
    corners1 = list(map(lambda coord: (round(coord[0], 0), round(coord[1], 0)), harris_corners))

    # adding the corners obtained from hough transform
    corners1 += hough_corners

    # removing the duplicate corners
    corners_no_dup = list(set(corners1))

    min_cell_width = w/30
    min_cell_height = h/30

    # grouping coordinates into probabale array that could fit a horizontal and vertical lien
    vertical_lines = group_lines(corners_no_dup,0,min_cell_height)
    horizontal_lines = group_lines(corners_no_dup,1,min_cell_height)

    actual_vertical_lines = fit_lines(vertical_lines)
    actual_horizontal_lines = fit_lines(horizontal_lines,is_horizontal=True)


    # Lines obtained from above method are not appropriate, we have to refine them

    x_probable = [i[1] for i in actual_horizontal_lines] # looking at the intercepts 
    y_probable = [i[1] for i in actual_vertical_lines]

    del_x_avg = average_distance(x_probable)   
    del_y_avg = average_distance(y_probable)  

    averaged_horizontal_lines1 = []         # This step here is fishy and needs refinement
    averaged_vertical_lines1 = []
    multiplier = 0.95
    i = 0
    while(1):
        averaged_horizontal_lines = average_out_similar_lines(actual_horizontal_lines,horizontal_lines,del_y_avg*multiplier,is_horizontal=True)
        averaged_vertical_lines = average_out_similar_lines(actual_vertical_lines,vertical_lines,del_x_avg*multiplier,is_horizontal=False)
        i += 1
        if(i >= 20 or len(averaged_horizontal_lines) == len(averaged_vertical_lines)):
            break
        else:
            multiplier -= 0.05

    averaged_horizontal_lines1 = average_out_similar_lines1(actual_horizontal_lines,horizontal_lines,del_y_avg*multiplier)
    averaged_vertical_lines1 = average_out_similar_lines1(actual_vertical_lines,vertical_lines,del_x_avg*multiplier)


    # plotting the lines to image to find the intersection points
    drawn_image6 = np.ones_like(cropped_image)*255
    for vx,vy,cx,cy in  averaged_horizontal_lines1 + averaged_vertical_lines1:
        w = cropped_image.shape[1]    
        cv2.line(drawn_image6, (int(cx-vx*w), int(cy-vy*w)), (int(cx+vx*w), int(cy+vy*w)), (0, 0, 255),1,cv2.LINE_AA)

    # -- Finding Intersection points -- 
    
    # Applying Harris Corner Detection to find the intersection points
    mesh_image = drawn_image6.copy()
    dst = cv2.cornerHarris(mesh_image,2,3,0.04)

    ret,dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    dst = cv2.dilate(dst,None)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TermCriteria_MAX_ITER,100,0.001)
    harris_corners = cv2.cornerSubPix(mesh_image,np.float32(centroids),(5,5),(-1,-1),criteria)
    drawn_image = cv2.cvtColor(drawn_image6, cv2.COLOR_GRAY2BGR)
    harris_corners = list(sorted(harris_corners[1:],key = lambda x : x[1]))

    # -- Finding out the grid color --


    grayscale = cropped_image.copy()
    # Perform adaptive thresholding to obtain binary image
    _, binary = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Perform morphological operations to remove small text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_ELLIPSE, kernel, iterations=1)

    # Invert the binary image
    inverted_binary = cv2.bitwise_not(binary)

    # Restore the image by blending the inverted binary image with the grayscale image
    restored_image = cv2.bitwise_or(inverted_binary, grayscale)

    # Apply morphological opening to remove small black dots
    kernel_opening = np.ones((3, 3), np.uint8)
    opened_image = cv2.morphologyEx(restored_image, cv2.MORPH_OPEN, kernel_opening, iterations=1)

    # Apply morphological closing to further refine the restored image
    kernel_closing = np.ones((5, 5), np.uint8)
    refined_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel_closing, iterations=1)

    # finding out the grid corner
    grid = []
    grid_nums = []
    across_clue_num = []
    down_clue_num = []

    sorted_corners = np.array(list(sorted(harris_corners,key=lambda x:x[1])))
    if(len(sorted_corners) == len(averaged_horizontal_lines1) * len(averaged_vertical_lines1)):
        sorted_corners_grouped = []
        for i in range(0,len(sorted_corners),len(averaged_vertical_lines1)):
            temp_arr = sorted_corners[i:i+len(averaged_vertical_lines1)]
            temp_arr = list(sorted(temp_arr,key=lambda x: x[0]))
            sorted_corners_grouped.append(temp_arr)

        for h_line_idx in range(0,len(sorted_corners_grouped)-1):
            for corner_idx in range(0,len(sorted_corners_grouped[h_line_idx])-1):
                # grabbing the four box coordinates
                box = [sorted_corners_grouped[h_line_idx][corner_idx],sorted_corners_grouped[h_line_idx][corner_idx+1],
                    sorted_corners_grouped[h_line_idx+1][corner_idx],sorted_corners_grouped[h_line_idx+1][corner_idx+1]]
                grid.append(get_square_color(refined_image,box))

        grid_formatted = []
        for i in range(0, len(grid), len(averaged_vertical_lines1) - 1):
            grid_formatted.append(grid[i:i + len(averaged_vertical_lines1) - 1])
        

        # if (x,y) is present in these array the cell (x,y) is already accounted as a part of answer of across or down
        in_horizontal = []
        in_vertical = []
        
        num = 0

        

        for x in range(0, len(averaged_vertical_lines1) - 1):
            for y in range(0, len(averaged_horizontal_lines1) - 1):

                # if the cell is black there's no need to number
                if grid_formatted[x][y] == '.':
                    grid_nums.append(0)
                    continue

                # if the cell is part of both horizontal and vertical cell then there's no need to number
                horizontal_presence = (x, y) in in_horizontal
                vertical_presence = (x, y) in in_vertical

                # present in both 1 1 
                if horizontal_presence and vertical_presence:
                    grid_nums.append(0)
                    continue

                # present in one i.e 1 0
                if not horizontal_presence and vertical_presence:
                    horizontal_length = 0
                    temp_horizontal_arr = []
                    # iterate in x direction until the end of the grid or until a black box is found
                    while x + horizontal_length < len(averaged_horizontal_lines1) - 1 and grid_formatted[x + horizontal_length][y] != '.':
                        temp_horizontal_arr.append((x + horizontal_length, y))
                        horizontal_length += 1
                    # if horizontal length is greater than 1, then append the temp_horizontal_arr to in_horizontal array
                    if horizontal_length > 1:
                        in_horizontal.extend(temp_horizontal_arr)
                        num += 1
                        across_clue_num.append(num)
                        grid_nums.append(num)
                        continue
                    grid_nums.append(0)
                # present in one 1 0        
                if not vertical_presence and horizontal_presence:
                    # do the same for vertical
                    vertical_length = 0
                    temp_vertical_arr = []
                    # iterate in y direction until the end of the grid or until a black box is found
                    while y + vertical_length < len(averaged_vertical_lines1) - 1 and grid_formatted[x][y+vertical_length] != '.':
                        temp_vertical_arr.append((x, y+vertical_length))
                        vertical_length += 1
                    # if vertical length is greater than 1, then append the temp_vertical_arr to in_vertical array
                    if vertical_length > 1:
                        in_vertical.extend(temp_vertical_arr)
                        num += 1
                        down_clue_num.append(num)
                        grid_nums.append(num)
                        continue
                    grid_nums.append(0)
                
                if(not horizontal_presence and not vertical_presence):

                    horizontal_length = 0
                    temp_horizontal_arr = []
                    # iterate in x direction until the end of the grid or until a black box is found
                    while x + horizontal_length < len(averaged_horizontal_lines1) - 1 and grid_formatted[x + horizontal_length][y] != '.':
                        temp_horizontal_arr.append((x + horizontal_length, y))
                        horizontal_length += 1
                    # if horizontal length is greater than 1, then append the temp_horizontal_arr to in_horizontal array
                                        
                    # do the same for vertical
                    vertical_length = 0
                    temp_vertical_arr = []
                    # iterate in y direction until the end of the grid or until a black box is found
                    while y + vertical_length < len(averaged_vertical_lines1) - 1 and grid_formatted[x][y+vertical_length] != '.':
                        temp_vertical_arr.append((x, y+vertical_length))
                        vertical_length += 1
                    # if vertical length is greater than 1, then append the temp_vertical_arr to in_vertical array
                    
                    if horizontal_length > 1 and horizontal_length > 1:
                        in_horizontal.extend(temp_horizontal_arr)
                        in_vertical.extend(temp_vertical_arr)
                        num += 1
                        across_clue_num.append(num)
                        down_clue_num.append(num)
                        grid_nums.append(num)
                    elif vertical_length > 1:
                        in_vertical.extend(temp_vertical_arr)
                        num += 1
                        down_clue_num.append(num)
                        grid_nums.append(num)
                    elif horizontal_length > 1:
                        in_horizontal.extend(temp_horizontal_arr)
                        num += 1
                        across_clue_num.append(num)
                        grid_nums.append(num)
                    else:
                        grid_nums.append(0)


    size = { 'rows' : len(averaged_horizontal_lines1)-1,
            'cols' : len(averaged_vertical_lines1)-1,
            }
    
    dict = {
        'size' : size,
        'grid' : grid,
        'gridnums': grid_nums,
        'across_nums': down_clue_num,
        'down_nums' : across_clue_num,
        'clues':{
            'across' : [],
            'down': []
        }
    }
    
    return dict

if __name__ == "__main__":
    img = cv2.imread("D:\\D\\Major Project files\\opencv\\movie.png",0)
    down = extract_grid(img)
    print(down)
    # img = Image.open("chalena3.jpg")
    # img_gray = img.convert("L")
    # print(extract_grid(img_gray))
