import cv2
import os 
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imutils
import random
from edge_detection import detect_quadrilaterals

# put in your path to github folder
to_github = r"C:\Users\morte\Documents\GitHub"
image_folder = to_github + "\ImageProcessing\Data\Test_2_1\Pos0"
dark_folder = to_github + "\ImageProcessing\Data\Test_2_dark\Pos0"

def plot_image(image, title, cmap = 'gray'):
    plt.figure(dpi=300)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


image_name1 = "img_000000001_Default_000.tif"
image_name2 = "img_000000000_Default_000.tif"
path1 = os.path.join(image_folder, image_name1)
path2 = os.path.join(image_folder, image_name2)

img1 = cv.imread(path1)
img2 = cv.imread(path2)

# equalize hist to enhance brightness in order to better find contours
img1_gray = cv.imread(path1,-1)
# x = 200
# y = 150
# img1_gray = img1_gray[256 - y: 256 + y, 320-x:320+x]
# img1_equ = cv.equalizeHist(img1_gray)

img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
img2_equ = cv.equalizeHist(img2_gray)

# plot_image(img1_gray, "Image 1")
# plot_image(img1_equ, "Image 1 equalized")
# plot_image(img2_gray, "Image 2")
# plot_image(img2_equ, "Image 2 equalized")

"""
a loop like this will show how the thresholded binary image looks like
for i in range(100):
    plt.imshow(cv.threshold(img1_equ, i, 255, 0)[1], cmap='gray')
    plt.title(i)
    plt.show()"""

""" canny work too then the loop looks like this
for i in range(100):
    plt.imshow(cv.Canny(img1_equ, i, i * 2), cmap='gray') 
    plt.title(i)
    plt.show()"""
# canny_threshold = 50
# canny_output = cv.Canny(img1_equ, canny_threshold, canny_threshold * 2) 
# canny_output = cv.Canny(img1_equ,60,150)
# contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

def find_biggest_contour(contours):
    return max(contours, key = cv2.contourArea)

def estimate_polygon(contour, precision = 0.01):
    epsilon = precision * cv.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True) #estimates a polygon on the biggest contour
    return corners

def find_biggest_square_quadrilateral(contours):
    counter = 0
    n = len(contours)
    while counter < n:
        c = contours[counter]
        e = estimate_polygon(c)
        if len(e) == 4:
            return e
        counter += 1
    print("No quadrilateral found")
    return None

def find_contours(image, threshold=2):
    ret, thresh = cv.threshold(image, threshold, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    return contours

def draw_quadrilateral(image):
    contours = find_contours(image, threshold=2)
    corners = find_biggest_square_quadrilateral(contours)
    cv.polylines(image, [corners], True, (0,0,255), 2)
    equalized = cv.equalizeHist(image)
    cv.imshow("image" , equalized)
    cv.waitKey(0)

def flip_square(s):
    index = np.argmin(np.sum(s,axis=1))
    s = np.roll(s, 4-index, axis = 0)
    return s
 
def polygon_area(xs, ys):
    """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon"""
    # https://stackoverflow.com/a/30408825/7128154
    return 0.5 * (np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))

def polygon_centroid(quad):
    """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon"""
    xs = list(quad[:,0,0])
    ys = list(quad[:,0,1])
    xy = np.array([xs, ys])
    c = np.dot(xy + np.roll(xy, 1, axis=1),
               xs * np.roll(ys, 1) - np.roll(xs, 1) * ys
               ) / (6 * polygon_area(xs, ys))
    return c

def contour_is_square(contour):
    if (len(contour) == 4) and (cv.contourArea(contour) > 200 ):
        # it needs to have some size since it also finds very tiny ones, which
        # cant be a cell
        return True
    else:
        return False
 
def find_corners(contour, precision = 0.04):
    epsilon = precision * cv.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True)
    return corners
    
for image in os.listdir(image_folder): # doesnt work at the moment
    break
    path = os.path.join(image_folder, image)
    img = cv.imread(path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    draw_quadrilateral(img_gray)
    
    contours = find_contours(img_gray)
    corners = find_biggest_square_quadrilateral(contours)
    
    x,y,w,h = cv2.boundingRect(corners)
    #dst = np.array([[x+w,y],[x,y],[x,y+h],[x+w,y+h]])
    dst = np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])
    s = corners.astype(np.float32)[:,0,:]
    s = flip_square(s)
    
    dst = dst.astype(np.float32)
    M = cv.getPerspectiveTransform(s, dst)
    warp = cv.warpPerspective(img_gray, M, (600,600))
    warp = warp[y:y+h, x:x+w]
    #cv2.rectangle(warp,(x,y),(x+w,y+h),(255,0,0),1)
    cv.imshow("image", warp*5)
    cv.waitKey(0)
    

def in_trust_region(quad, x1,y1,x2,y2):
    # find out if quad is inside trust region
    xs = quad[:,0,0]
    ys = quad[:,0,1]
    if (sum(ys > y1) == 4) and (sum(ys < y2) == 4):
        if (sum(xs > x1) == 4) and (sum(xs < x2) == 4):
            return True
    else:
        return False

def quads_in_trust_region(quadrilaterals, trust_region):
    # return list of quads in trust region
    x1,y1 = trust_region[0]
    x2,y2 = trust_region[1]
    quadrilaterals = list(filter(lambda quad: in_trust_region(quad, x1,y1,x2,y2), quadrilaterals))
    return quadrilaterals

def estimate_polygons(contours):
    # estimate polygons based on contours
    return list(map(find_corners, contours))

def keep_quadrilaterals(polygons):
    # filter out polygons not a quadrilateral
    return list(filter(contour_is_square, polygons))

# get random color
color = lambda : (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def closest_center(center, centers):
    # find closest point in centers to given center
    #https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
    centers = np.asarray(centers)
    deltas = centers - center
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def transform_quad(quad,image):
    # makes a quad a square based on the bounding rectangle around it 
    # the dimensions is passed on so another frame showing the same quad
    # will get the same dimensions
    x,y,w,h = cv2.boundingRect(quad)
    dst = np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]]).astype(np.float32)
    quad = quad.astype(np.float32)[:,0,:]
    quad = flip_square(quad)
    M = cv.getPerspectiveTransform(quad, dst)
    warp = cv.warpPerspective(image, M, (600,600)) # these numbers??
    return warp[y:y+h, x:x+w], x,y,w,h

def transform_quad_to_dim(quad, x,y, w, h, image):
    # transform quad to square given dimension found in previous function
    quad = quad.astype(np.float32)[:,0,:]
    dst = np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]]).astype(np.float32)
    quad = flip_square(quad)
    M = cv.getPerspectiveTransform(quad, dst)
    warp = cv.warpPerspective(image, M, (600,600))
    return warp[y:y+h, x:x+w]    



input_image = img1_gray
input_image = cv2.blur(input_image,(3,3))
# set values < 0 to 0

# plot_image(input_image, "hello")
# contours = cv.findContours(input_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
# contours_polygons = estimate_polygons(contours)
# quadrilaterals = keep_quadrilaterals(contours_polygons)
quadrilaterals = detect_quadrilaterals(img1_gray)
cv.drawContours(img1, quadrilaterals, -1, (255,0,0),3) 
cv.imshow('ger', img1*5)
cv2.waitKey(0)
cv2.destroyAllWindows()

#### the next needs to be generalized but tracks cells from the first to second image ###
i,j = 11,1
th = cv.adaptiveThreshold(img1_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,i,j)
contours = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
contours_polygons = estimate_polygons(contours)
quadrilaterals = keep_quadrilaterals(contours_polygons)

x = 95
y = 100
# trust region is two points, upper left and lower right
trust_region = (320-x,256-y),(320+x,256+y)
quadrilaterals_tr = quads_in_trust_region(quadrilaterals, trust_region)
centers = list(map(polygon_centroid, quadrilaterals_tr))

time=0
quads=[]
# at time zero we have no quads so we initialize quads to keep track of it
for i, center in enumerate(centers):
    quad = dict()
    quad['centers'] = [center] # list of centers as they move due to the camera movement
    quad['color'] = color() # random color assigned
    save, x,y,w,h = transform_quad(quadrilaterals_tr[i], img1_gray)
    quad['dimensions'] = (x,y,w,h) # save dimensions which will be used in future loops
    quad[time] = save # at time 0 we save the image
    cv.circle(img1, tuple(map(int, center)), 2, quad['color'], 2)
    quads.append(quad)
cv.rectangle(img1, trust_region[0], trust_region[1], (255,255,0), 2)


time += 1
#cv.drawContours(img1, contours, -1, (0,255,0),3) 
cv.drawContours(img1, quadrilaterals_tr, -1, (255,0,0),3) 
cv.imshow('ger', img1*5)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv.rectangle(img2, trust_region[0], trust_region[1], (255,255,0), 2)

# next frame, find quads again
i,j = 11,1
th2 = cv.adaptiveThreshold(img2_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,i,j)
contours2 = cv.findContours(th2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
contours_polygons2 = estimate_polygons(contours2)
quadrilaterals2 = keep_quadrilaterals(contours_polygons2)
quadrilaterals_tr2 = quads_in_trust_region(quadrilaterals2, trust_region)
centers2 = list(map(polygon_centroid, quadrilaterals_tr2))

n1, n2 = len(centers), len(centers2)

def center_center_avg_distance(from_centers, to_centers, indexes):
    '''
    Will find the average center-center distance between frames. Specifically,
    the average distance between the centers in from_centers to the centers
    in to_center matched by indexes. Indexes then shows what centers in to_centers
    were the same as in from_centers.
    '''
    
    from_centers = np.asanyarray(from_centers)
    to_centers = np.asanyarray(to_centers)
    to_centers = to_centers[indexes]
    x_avg = np.average(from_centers[:,0]) - np.average(to_centers[:,0])
    y_avg = np.average(from_centers[:,1]) - np.average(to_centers[:,1])
    return x_avg, y_avg

if n1 == n2:
    # the same centers are showing in trust region, so they are the same as before
    # relate each to the closest center from before
    center_indexes = list(map(lambda x: closest_center(x, centers), centers2))
    for i, center_index in enumerate(center_indexes):
        #closest_center_index = closest_center(center, centers)
        quads[center_index]['centers'].append(centers2[i])
        x,y,w,h = quads[center_index]['dimensions']
        save = transform_quad_to_dim(quadrilaterals_tr2[i], x, y, w, h, img2_gray)
        quads[center_index][time] = save
        cv.circle(img2, tuple(map(int, (centers2[i]))), 2, quads[center_index]['color'], 2)    
        
elif n1 > n2:
    k = n1 - n2 
    # k quads have gone out of "focus"
    # n2 centers from before remain, find out which they respond to by minimum distance
    center_indexes = list(map(lambda x: closest_center(x, centers), centers2))
    for i, center_index in enumerate(center_indexes):
        quads[center_index]['centers'].append(centers2[i])
        x,y,w,h = quads[center_index]['dimensions']
        save = transform_quad_to_dim(quadrilaterals_tr2[i], x, y, w, h, img2_gray)
        quads[center_index][time] = save
        cv.circle(img2, tuple(map(int, quads[center_index]['centers'][time])), 2, quads[center_index]['color'], 2)
    # updating the center positions of the k out of focus quads
    avg_distance = center_center_avg_distance(centers2, centers, center_indexes)
    gone_centers_indexes = list(set(range(n1)).difference(center_indexes))
    for center_index in gone_centers_indexes:
        old_center = quads[center_index]['centers'][-1]
        new_center = old_center + avg_distance
        quads[center_index]['centers'].append(new_center)
        quads[center_index][time] = np.nan
        cv.circle(img2, tuple(map(int, quads[center_index]['centers'][time])), 2, quads[center_index]['color'], 2)

else:
    k = n2 - n1
    # n1 < n2: 
    # k new centers are showing
    # find out which n1 centers respond to those we had before by minimum distance
    # the remaining k are new (or maybe seen before yikes)
    center_indexes = list(map(lambda x: closest_center(x, centers2), centers))
    for i, center_index in enumerate(center_indexes):
        quads[i]['centers'].append(centers2[center_index])
        x,y,w,h = quads[i]['dimensions']
        save = transform_quad_to_dim(quadrilaterals_tr2[center_index], x, y, w, h, img2_gray)
        quads[i][time] = save
        cv.circle(img2, tuple(map(int, quads[i]['centers'][time])), 2, quads[i]['color'], 2)
    # k new centers
    new_centers_indexes = list(set(range(n2)).difference(center_indexes))
    #new_centers = [centers2[i] for i in new_centers_indexes]
    for i, center_index in enumerate(new_centers_indexes):
        print(i, center_index)
        quad = dict()
        quad['centers'] = [centers2[center_index]]
        quad['color'] = color() # random color assigned
        save, x,y,w,h = transform_quad(quadrilaterals_tr2[center_index], img2_gray)
        quad['dimensions'] = (x,y,w,h)
        quad[time] = save
        cv.circle(img2, tuple(map(int, quad['centers'][0])), 2, quad['color'], 2)
        quads.append(quad)
cv.drawContours(img2, quadrilaterals_tr2, -1, (255,0,0),3) 
cv.imshow('ger', img2*5)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.polylines(img1, [corners], True, (0,0,255), 2)
# the polygon to its edges see doc










cv2.imshow('Contours', img1*5)
cv.imshow('ger', img1_gray*5)
cv2.waitKey(0)
cv2.destroyAllWindows()


threshold = 2
ret, thresh = cv.threshold(img2_equ, threshold, 255, 0) # doesnt quite work
ret, thresh = cv.threshold(img2_gray, threshold, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# contours = cv.findContours(img2_equ, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
# cv.drawContours(img2, contours, -1, (255,0,0),3) 
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
corners = find_biggest_square_quadrilateral(contours)
#cv.drawContours(img2, contours, -1, (255,0,0),3) 
cv2.polylines(img2, [corners], True, (0,0,255), 2)

# biggest_contour = find_biggest_contour(contours)

# corners = estimate_quadrilateral(biggest_contour)
# cv2.polylines(img2, [corners], True, (0,255,0), 2)
cv2.imshow('Contours', img2*5)

cv2.waitKey(0)
cv2.destroyAllWindows()"""