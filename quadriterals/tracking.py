import cv2
import os 
import numpy as np
import cv2 as cv
import random
from edge_detection import detect_quadrilaterals
from preprocess_images import normalize_images
import matplotlib.pyplot as plt
raw_images = r"C:\Users\morte\Desktop\Test_4_1\Pos0"
save_folder = r"C:\Users\morte\Desktop\Test_4_1\pre-processed"

# normalize_images(raw_images, save_folder) #run once
image_folder = save_folder

## Trust region
x = 250
y = 125
# trust region is two points, upper left and lower right
trust_region = (320-x,256-y),(320+x,256+y)

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

color = lambda : (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def flip_square(s):
    '''
    flips a quad so the left upper corner is first in the array, and then the
    points are counter clockwise from that.

    '''
    index = np.argmin(np.sum(s[:,0,:],axis=1))
    s = np.roll(s, 4-index, axis = 0)
    return s
 
def transform_quad(quad, image, enlarge = False):
    # makes a quad a square based on the bounding rectangle around it 
    # the dimensions is passed on so another frame showing the same quad
    # will get the same dimensions
    quad = flip_square(quad)
    if enlarge:
        k = 3
        addition = np.array([[[-k, -k]],
                             [[-k,  k]],
                             [[ k,  k]],
                             [[ k, -k]]])
        quad = quad + addition
    x,y,w,h = cv2.boundingRect(quad)
    dst = np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]]).astype(np.float32)
    quad = quad.astype(np.float32)[:,0,:]
    # quad = flip_square(quad)
    M = cv.getPerspectiveTransform(quad, dst)
    warp = cv.warpPerspective(image, M, (600,600)) # these numbers??
    return warp[y:y+h, x:x+w], x,y,w,h

def transform_quad_to_dim(quad, x,y, w, h, image, enlarge = False):
    quad = flip_square(quad)
    # transform quad to square given dimension found in previous function
    if enlarge:
        k = 3
        addition = np.array([[[-k, -k]],
                             [[-k,  k]],
                             [[ k,  k]],
                             [[ k, -k]]])
        quad = quad + addition
    quad = quad.astype(np.float32)[:,0,:]
    dst = np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]]).astype(np.float32)
    # quad = flip_square(quad)
    M = cv.getPerspectiveTransform(quad, dst)
    warp = cv.warpPerspective(image, M, (600,600))
    return warp[y:y+h, x:x+w]    


def closest_center(center, centers):
    # find closest point in centers to given center
    #https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
    centers = np.asarray(centers)
    deltas = centers - center
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    argmin = np.argmin(dist_2)
    return argmin, dist_2[argmin]


# keep all centers seen in a single list  
# have the index correspond to a quad
# when we see new centers, add them         
centers = [np.array([336, 283]), np.array([335, 193])]
centers_new = [np.array([335, 193]), np.array([336, 283])]

def center_center_avg_distance(centers_all, centers_new, indexes, distance):
    '''
    Will find the average center-center distance between frames. Specifically,
    the average distance between the centers in from_centers to the centers
    in to_center matched by indexes. Indexes then shows what centers in to_centers
    were the same as in from_centers.
    '''
    centers_new = np.asanyarray(centers_new)
    from_centers = centers_all[indexes[:,0]]
    to_centers = centers_new[indexes[:,1]]
    if (len(from_centers)==0) or (len(to_centers)==0):
        return distance
    x_avg = np.average(to_centers[:,0]) - np.average(from_centers[:,0])
    y_avg = np.average(to_centers[:,1]) - np.average(from_centers[:,1]) 
    return x_avg, y_avg

def center_matching(centers_all, centers_new):
    '''
    for each center in centers_new, the function returns the index of the center 
    in center_all which is closest to the center in centers_new.
    If there are more centers in new than all, the leftover is returned. 
    This is also the case if the point is too far from the minimum.
    '''
    used = []
    matches = np.empty((0,2),int)
    n2 = len(centers_new)
    # matches = []
    new = []
    for i in range(n2):
        match, distance = closest_center(centers_new[i], centers_all)
        print("distance",distance)
        print("match",match)
        if distance < 3000:
            matches = np.vstack((matches, [match, i]))
            used.append(matches)
            print(matches)
            # [index in centers_all, index in centers_new]
        else:
            new.append(i)
    
    return matches, new

def grid_placement(point1, point2):
    '''
    
    '''
    x1, y1 = point1
    x2, y2 = point2
    print(point1,point2)
    if (abs(x2-x1) < 10):
        if y1 - y2 > 0:
            return 'down'
        else:
            return 'up'
    else:
        if x1 - x2 < 0:
            return 'right'
        else:
            return 'left'

        
# LOOP #
time=0
quads=[]
#centers = np.zeros((50,50), dtype='i,i')
color = lambda : (random.randint(0, 2**16), random.randint(0, 2**16), random.randint(0, 2**16))
colors = [color() for x in range(1000)]
centers_all = np.empty((0,2))
avg_distance = 0
N = 100
quads = [ [] for _ in range(N) ]
dimensions = []
quads_counter = 0

for image_name in os.listdir(image_folder)[:]:
    if '.tif' not in image_name:
        continue
    print("Image name:",image_name)
    path = os.path.join(image_folder, image_name)
    img_gray = cv.imread(path, -1)
    img = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    img_gray8 = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2GRAY)
    quadrilaterals = detect_quadrilaterals(img_gray8) #, alpha = 5, opening_iterations=2,
                                           #window_size=15, k = 32, offset=20)
    quadrilaterals_tr = quads_in_trust_region(quadrilaterals, trust_region) 
    
    if time == 0:
        centers_old = list(map(polygon_centroid, quadrilaterals_tr))
        print("Centers_old", centers_old)
        if len(centers_old) == 0:
            continue # wait until something is detected to start
            
        for i, center in enumerate(centers_old):
            # centers_all.append(center)
            # print("hello",center)
            centers_all = np.vstack((centers_all, np.array([center])))
            save, x,y,w,h = transform_quad(quadrilaterals_tr[i], img_gray, enlarge=True)
            quads[i].append(save)
            dimensions.append([x,y,w,h])
            quads_counter += 1
            
    
    else:
        centers_new = list(map(polygon_centroid, quadrilaterals_tr))
        if len(centers_new) == 0: # if nothing was detected, continue the iteration
            centers_old = []
            time += 1
            continue
        
        n1, n2 = len(centers_old), len(centers_new)
        index_in_center_all, new = center_matching(centers_all, centers_new)
        # print("n1:",n1,"n2:",n2)
        # print("centers_all", centers_all)
        # print("center_new", centers_new)
        # print("")
        # print("index_in_center_all:\n", index_in_center_all)
        # print("")
        # print("new:", new)
        # print("n1:",n1,"n2:",n2)
        
        avg_distance = center_center_avg_distance(centers_all, centers_new, index_in_center_all, avg_distance)
        in_focus = []
        
        for i, j in index_in_center_all:
            centers_all[i] = centers_new[j]
            x, y, w, h = dimensions[i]
            save = transform_quad_to_dim(quadrilaterals_tr[j], x, y, w, h, img_gray, enlarge=True)
            quads[i].append(save)
            in_focus.append(i)
        for i in new:
            # centers_all.append(centers_new[i])
            quads_counter += 1
            centers_all = np.vstack((centers_all, centers_new[i]))
            save, x,y,w,h = transform_quad(quadrilaterals_tr[i], img_gray, enlarge=True)
            quads[quads_counter].append(save)
            dimensions.append([x,y,w,h])
            
        out_of_focus = set(range(quads_counter)).difference(in_focus)
        print("avg_distance",avg_distance)
        for center_index in out_of_focus:
            centers_all[center_index] += avg_distance
            quads[center_index].append(np.nan)
        
        
        centers_old = centers_new
        
        
    time += 1
    
    print("final",centers_all)
    print("len", len(centers_all))
    # plt.scatter(centers_all[:,0], centers_all[:,1])
    # plt.show()
    for i, center in enumerate(centers_all):
        cv.circle(img, tuple(map(int, (center))), 2, colors[i], 2)    
    cv.rectangle(img, trust_region[0], trust_region[1], (255,255,0), 2)
    cv.drawContours(img, quadrilaterals_tr, -1, (2**16,0,0),3) 
    cv.imshow("Image", img)
    cv2.waitKey(1)
    
cv2.destroyAllWindows()

# grid = np.array(quads[0])
# point_old = centers_all[0]
# for point in centers_all[1:]:
#     print(grid_placement(point_old, point))
#     point_old = point
        
"""
for image_name in os.listdir(image_folder):
    
    print("Image name:",image_name)
    path = os.path.join(image_folder, image_name)
    img_gray = cv.imread(path, -1)
    
    quadrilaterals = detect_quadrilaterals(img_gray)
    quadrilaterals_tr = quads_in_trust_region(quadrilaterals, trust_region) 
    if time == 0:
        centers_old = list(map(polygon_centroid, quadrilaterals_tr))
        if len(centers_old) == 0:
            continue # wait until something is detected to start
        for i, center in enumerate(centers_old):
            quad = dict()
            centers_all.append(center)
            quad['centers'] = [center] # list of centers as they move due to the camera movement
            quad['color'] = color() # random color assigned
            save, x,y,w,h = transform_quad(quadrilaterals_tr[i], img_gray)
            quad['dimensions'] = (x,y,w,h) # save dimensions which will be used in future loops
            quad[time] = save # at time 0 we save the image
            quads.append(quad)
            
    else:
        centers_new = list(map(polygon_centroid, quadrilaterals_tr))
        if len(centers_new) == 0: # if nothing was detected, continue the iteration
            centers_old = []
            time += 1
            continue
        
        n1, n2 = len(centers_old), len(centers_new)
        
        if n1 == n2:
            print(time, "n1 == n2")
            # the same centers are showing in trust region, so they are the same as before
            # relate each to the closest center from before
            center_indexes = list(map(lambda x: closest_center(x, centers_old), centers_new))
            for i, center_index in enumerate(center_indexes):
                #closest_center_index = closest_center(center, centers)
                quads[center_index]['centers'].append(centers_new[i])
                x,y,w,h = quads[center_index]['dimensions']
                save = transform_quad_to_dim(quadrilaterals_tr[i], x, y, w, h, img_gray)
                quads[center_index][time] = save
                #cv.circle(img2, tuple(map(int, (centers_new[i]))), 2, quads[center_index]['color'], 2)    
    
        elif n1 > n2:
            print(time, "n1 > n2")
            k = n1 - n2 
            # k quads have gone out of "focus"
            # n2 centers from before remain, find out which they respond to by minimum distance
            center_indexes = list(map(lambda x: closest_center(x, centers_old), centers_new))
            for i, center_index in enumerate(center_indexes):
                quads[center_index]['centers'].append(centers_new[i])
                x,y,w,h = quads[center_index]['dimensions']
                save = transform_quad_to_dim(quadrilaterals_tr[i], x, y, w, h, img_gray)
                quads[center_index][time] = save
                #cv.circle(img_gray, tuple(map(int, quads[center_index]['centers'][time])), 2, quads[center_index]['color'], 2)
            # updating the center positions of the k out of focus quads
            avg_distance = center_center_avg_distance(centers_new, centers_old, center_indexes)
            gone_centers_indexes = list(set(range(n1)).difference(center_indexes))
            for center_index in gone_centers_indexes:
                old_center = quads[center_index]['centers'][-1]
                new_center = old_center + avg_distance
                quads[center_index]['centers'].append(new_center)
                quads[center_index][time] = np.nan
                #cv.circle(img_gray, tuple(map(int, quads[center_index]['centers'][time])), 2, quads[center_index]['color'], 2)

        else:
            print(time, "n1 < n2")
            k = n2 - n1
            # n1 < n2: 
            # k new centers are showing
            # find out which n1 centers respond to those we had before by minimum distance
            # the remaining k are new (or maybe seen before yikes)
            center_indexes = list(map(lambda x: closest_center(x, centers_new), centers_old))
            for i, center_index in enumerate(center_indexes):
                quads[i]['centers'].append(centers_new[center_index])
                x,y,w,h = quads[i]['dimensions']
                save = transform_quad_to_dim(quadrilaterals_tr[center_index], x, y, w, h, img_gray)
                quads[i][time] = save
                # cv.circle(img2, tuple(map(int, quads[i]['centers'][time])), 2, quads[i]['color'], 2)
            # k new centers
            new_centers_indexes = list(set(range(n2)).difference(center_indexes))
            #new_centers = [centers2[i] for i in new_centers_indexes]
            for i, center_index in enumerate(new_centers_indexes):
                quad = dict()
                quad['centers'] = [centers_new[center_index]]
                quad['color'] = color() # random color assigned
                save, x,y,w,h = transform_quad(quadrilaterals_tr[center_index], img_gray)
                quad['dimensions'] = (x,y,w,h)
                quad[time] = save
                # cv.circle(img2, tuple(map(int, quad['centers'][0])), 2, quad['color'], 2)
                quads.append(quad)
    
        centers_old = centers_new
    cv.rectangle(img_gray, trust_region[0], trust_region[1], (255,255,0), 2)
    cv.drawContours(img_gray, quadrilaterals_tr, -1, (255,0,0),3) 
    cv.imshow("Image", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    time += 1
    
"""    
    