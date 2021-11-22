#!/usr/bin/env python3
'''
## interseis_lib.py

Library of python functions to be used with interseismic_practical.ipynb.

'''

# packages
import subprocess as subp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as path
import warnings

#-------------------------------------------------------------------------------

def screw_disc(x, s, d, c, xc=0):
    '''
    Function to calculate displacements/velocities due to slip on a deep 
    screw dislocation (infinitely long strike slip fault. 
    After Savage and Burford (1973).
    v = (s/pi)*arctan((x+xc)/d)

    INPUTS
        x = vector of distances from fault
        s = slip or slip rate on deep dislocation
        d = locking depth [same units as x]
        c = scalar offset in y [same unit as s]
        xc = offset to fault location [same units as x]
    OUTPUTS
        v = vector of displacements or velocities at locations defined by x
          [same units as s]
          
    USEAGE:
        v = deepdisloc(x, s, d)

    '''
    
    v = (s/np.pi) * np.arctan(x/d) + c
    
    return v

#-------------------------------------------------------------------------------

def fault_creep(x, s1, s2, d1, d2, c, xc=0):
    '''
    Model for interseismic strain accumulation and fault creep.
    Taken from Hussain et al. (2016).

    INPUTS
        x = vector of distances from fault
        s1 = slip or slip rate on deep dislocation
        s2 = creep or creep rate
        d1 = locking depth
        d2 = depth to bottom of creeping section
        c = scalar offset in y [same unit as s]
        xc = offset of fault location
    OUTPUTS
        v = vector of displacements or velocities at locations defined by x
          [same units as s]

    '''
        
    v = (s1/np.pi)*np.arctan((x+xc)/d1) + c - s2*((1/np.pi)*np.arctan((x+xc)/d2) + (x+xc<=0)*0.5 - (x+xc>0)*0.5)
        
    return v

#-------------------------------------------------------------------------------

def gen_inc(inc_min, inc_max, heading, x, y):
    '''
    Generates a synthetic incidence angle grid based on a heading and a minimum 
    and maximum incidence angle.
    
    INPUTS
        inc_min = minimum desired incidence angle
        inc_max = maximum desired incidence angle
    OUTPUTS
        inc_grid = numpy array of incidence angles
    '''
    
    # if value is negative, add 360 until its positive
    while heading < 0:
        heading = heading + 360
        
    # if heading is over 360, subtract 360 until within range
    while heading > 360:
        heading = heading - 360
    
    # calculate the length of the look direction across the grid
    if (0 <= heading < 45) or (315 <= heading <= 360):
        d = (np.amax(x) - np.amin(x)) / np.cos(np.deg2rad(heading))
    elif (45 <= heading < 135):
        d = (np.amax(y) - np.amin(y)) / np.sin(np.deg2rad(heading))
    elif (135 <= heading < 225):
        d = - (np.amax(x) - np.amin(x)) / np.cos(np.deg2rad(heading))
    elif (225 <= heading < 315):
        d = - (np.amax(y) - np.amin(y)) / np.sin(np.deg2rad(heading))
    
    # calculate the gradient of incidence angles to get from inc_min to inc_max
    m = (inc_max-inc_min) / d
    
    # split this gradient into x and y components
    mx = m * np.cos(np.deg2rad(heading))
    my = m * np.sin(np.deg2rad(heading))
    
    # apply gradients to grid using 1st order 2D polynomial
    xx, yy = np.meshgrid(x,y)
    inc_grid = mx*xx + my*yy + (inc_min + (inc_max-inc_min)/2)
    
    return inc_grid

#-------------------------------------------------------------------------------

def likelihood(x, v, m, W):
    '''
    Likelihood function for monte carlo.
    
    INPUTS
        x = vector of distances from fault
        v = velocities at locations defined by x
        m = model parameters, [0] = slip (mm/yr), [1] = locking depth (km), [2] = scalar offset (mm/yr)
        W = weight matrix (inverse of the VCM)
    OUTPUTS
        ll = value of the loglikelihood function
    '''
    
    ll = np.nansum((np.transpose(v-screw_disc(x, m[0], m[1], m[2]))*W*(v-screw_disc(x, m[0], m[1], m[2]))));
    
    return ll

#-------------------------------------------------------------------------------

def prior(m,m_min,m_max):
    '''
    Log prior function for monte carlo.
    
    INPUTS
        m = model values
        m_min = lower model limits
        m_max = upper model limits
    OUTPUTS
        lp = true if within limits, false if any aren't
    '''
    
    lp = np.all(np.all(m>=m_min) & np.all(m<=m_max))
    
    return lp

#-------------------------------------------------------------------------------

def rms_misfit(a,b):
    '''
    Calculate the root-mean-square misfit between 'a' and 'b'.
    
    INPUTS
        a,b = two arrays of same length
    OUTPUTS
        rms = rms misfit between a and b (a-b)
    '''
    
    rms = np.sqrt(np.nanmean((a-b)**2))
    
    return rms

#-------------------------------------------------------------------------------

def get_par(par_file,par_name):
    '''
    Returns the value of the requested parameter in the parameter file.
    
    INPUTS
        par_file = name of param file (str)
        par_name = name of desired par (str)
    OUTPUTS
        par_val = value of param for par file
    '''
    
    with open(par_file, 'r') as f:
        for line in f.readlines():
            if par_name in line:
                par_val = line.split()[1].strip()
    
    return par_val

#-------------------------------------------------------------------------------

def profile_data(x,y,data,prof_start,prof_end,params):
    
    '''
    Generates a profile through gridded data, returning both projected data points
    and binned means.
    
    INPUTS:
    data = numpy array of values to profile
    x = vector of coords for the x axis
    y = vector of coords for the y axis
    prof_start = (x, y) pair for the start of the profile line
    prof_end = (x, y) pair for the end of the profile line
    params = dictionary of parameters for the profiler (currently nbins and width)
    
    '''
    
    xx,yy = np.meshgrid(x,y)
    
    prof_start = np.array(prof_start)
    prof_end = np.array(prof_end)
    
    # Profile dimensions relative to profile itself
    prof_dist = np.sqrt((prof_start[1]-prof_end[1])**2 + (prof_start[0]-prof_end[0])**2)
    prof_bin_edges = np.linspace(0, prof_dist ,params["nbins"]+1)    
    prof_bin_mids = (prof_bin_edges[:-1] + prof_bin_edges[1:]) / 2
    
    # Profile points in lat long space
    bin_mids = np.linspace(0,1,params["nbins"]+1)
    bin_grad = prof_end - prof_start
    x_mids = prof_start[0] + (bin_mids * bin_grad[0])
    y_mids = prof_start[1] + (bin_mids * bin_grad[1])
    
    # Gradient of line perpendicular to profile
    bin_grad_norm = (params["width"]/2) * bin_grad / np.linalg.norm(bin_grad)
    
    # Corner points of bins
    bin_x1 = x_mids + bin_grad_norm[1]
    bin_x2 = x_mids - bin_grad_norm[1]
    bin_y1 = y_mids - bin_grad_norm[0]
    bin_y2 = y_mids + bin_grad_norm[0]
    
    # Pre-allocate outputs
    bin_val = np.zeros_like((bin_x1[:-1]))
    bin_std = np.zeros_like(bin_val)
    
    # Trim data set to points inside any bin (improves run time)
    full_poly = path.Path([(bin_x1[0], bin_y1[0]), (bin_x1[-1], bin_y1[-1]), (bin_x2[-1], bin_y2[-1]), (bin_x2[0], bin_y2[0])])
    poly_points = full_poly.contains_points(np.transpose([xx.flatten(),yy.flatten()]))
    poly_points = poly_points.reshape(data.shape)
    trim_data = data[poly_points]
    trim_xx = xx[poly_points]
    trim_yy = yy[poly_points]
    
    # Loop through each bin identifying the points that they contain
    for ii in range(0,params["nbins"]):
                            
        poly_x = np.array([bin_x1[ii], bin_x1[ii+1], bin_x2[ii+1], bin_x2[ii]]);
        poly_y = np.array([bin_y1[ii], bin_y1[ii+1], bin_y2[ii+1], bin_y2[ii]]);
        
        poly = path.Path([(poly_x[0], poly_y[0]), (poly_x[1], poly_y[1]), (poly_x[2], poly_y[2]), (poly_x[3], poly_y[3])])
        
        poly_points = poly.contains_points(np.transpose([trim_xx,trim_yy]))
                            
        in_poly_vals = trim_data[poly_points]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bin_val[ii] = np.nanmean(in_poly_vals)
    
    # get point cloud
    poly_x = np.array([bin_x1[0], bin_x1[-1], bin_x2[-1], bin_x2[0]])
    poly_y = np.array([bin_y1[0], bin_y1[-1], bin_y2[-1], bin_y2[0]])
    points_poly = np.vstack((poly_x,poly_y)).T
    points_poly = np.vstack((points_poly,np.array([points_poly[0,0],points_poly[0,1]])))
    
    poly = path.Path([(poly_x[0], poly_y[0]), (poly_x[1], poly_y[1]), (poly_x[2], poly_y[2]), (poly_x[3], poly_y[3])])
    poly_points = poly.contains_points(np.transpose([trim_xx,trim_yy]))
    points_val = trim_data[poly_points]
    points_x = trim_xx[poly_points]
    points_y = trim_yy[poly_points]
    
    prof_m = (prof_start[1] - prof_end[1]) / (prof_start[0] - prof_end[0])
    points_m = (prof_start[1] - points_y) / (prof_start[0] - points_x)
    points_prof_angle = np.arctan((points_m - prof_m) / (1 + prof_m * points_m))
    points2prof_start = np.sqrt((prof_start[1] - points_y)**2 + (prof_start[0] - points_x)**2)
    points_dist = points2prof_start * np.cos(points_prof_angle)
    
    return bin_val, prof_bin_mids, points_val, points_dist, points_poly

#-------------------------------------------------------------------------------

def profile_fault_intersection(prof_start,prof_end,fault_trace):
    '''
    Calculates the distance along a profile at which it intersects a fault, and 
    the angle between the two at this point.
    
    INPUTS:
        prof_start = (x,y) coord
        prof_end = (x,y) coords
        fault_trace = x and y coords of fault trace
        
    OUTPUTS:
        intersection_distance
        intersection_angle
    '''
    
    # loop through all fault segments to find intersection
    for ind in range(fault_trace.shape[0]-1):
        if intersect(prof_start,prof_end,fault_trace[ind,:],fault_trace[ind+1,:]):
            inter_ind = ind
            break
    
    # get coords for either side of intersection segment
    fault_inter_coords = fault_trace[inter_ind:inter_ind+2,:]
    
    # calc gradient of of profile line and fault segment
    prof_m = (prof_start[1] - prof_end[1]) / (prof_start[0] - prof_end[0])
    fault_m = (fault_inter_coords[0,1] - fault_inter_coords[1,1]) / (fault_inter_coords[0,0] - fault_inter_coords[1,0])
    
    # calculate intersection angle
    intersection_angle = np.arctan((fault_m - prof_m) / (1 + prof_m * fault_m));
    
    # calculate distance to intersection point (from https://stackoverflow.com/questions/3252194/numpy-and-line-intersections)
    s = np.vstack([prof_start,prof_end,fault_inter_coords[0,:],fault_inter_coords[1,:]])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    intersection_point = np.array([x/z, y/z])
    intersection_distance = np.sqrt((intersection_point[0]-prof_start[0])**2 + (intersection_point[1]-prof_start[1])**2);    
    
    return intersection_distance, np.rad2deg(intersection_angle)


def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)