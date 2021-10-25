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

#-------------------------------------------------------------------------------

def screw_disc(x, s, d, c):
    '''
    Function to calculate displacements/velocities due to slip on a deep 
    screw dislocation (infinitely long strike slip fault. 
    After Savage and Burford (1973).
    v = (s/pi)*arctan(x/d)

    INPUTS
        x = vector of distances from fault
        s = slip or slip rate on deep dislocation
        d = locking depth [same units as x]
        c = scalar offset in y [same unit as s]
    OUTPUTS
        v = vector of displacements or velocities at locations defined by x
          [same units as s]
          
    USEAGE:
        v = deepdisloc(x, s, d)

    '''
    
    v = (s/np.pi) * np.arctan(x/d) + c
    
    return v


#-------------------------------------------------------------------------------

def loglike(x, v, m, W):
    '''
    INPUTS
        x = vector of distances from fault
        v = velocities at locations defined by x
        m = model parameters, [0] = slip (mm/yr), [1] = locking depth (km), [2] = scalar offset (mm/yr)
        W = weight matrix (inverse of the VCM)
    OUTPUTS
        ll = value of the loglikelihood function
    '''
    
    m = m * np.array([0.001, 1000, 0.001])
    
    #ll = np.sum((np.transpose(v-screw_disc(x, m[0], m[1], m[2]))*W*(v-screw_disc(x, m[0], m[1], m[2]))));
    ll = np.sum(0.0005*(np.transpose(v-screw_disc(x, m[0], m[1], m[2]))*W*(v-screw_disc(x, m[0], m[1], m[2]))));
    
    return ll

#-------------------------------------------------------------------------------

def logprior(m,m_min,m_max):
    '''
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
    INPUTS
        a,b = two arrays of same length
    OUTPUTS
        rms = rms misfit between a and b (a-b)
    '''
    
    rms = np.sqrt(np.mean((a-b)**2))
    
    return rms

#-------------------------------------------------------------------------------

def get_par(par_file,par_name):
    '''
    INPUTS
        par_file = name of param file (str)
        par_name = name of desired par (str)
    OUTPUTS
        par_val = value of param for par file
    '''
    
    par_val = subp.check_output(['grep', par_name, par_file]).decode().split()[1].strip()
    return par_val

#-------------------------------------------------------------------------------

def profile_data(x,y,data,prof_start,prof_end,params):
    
    '''
    Generates a profile through gridded data.
    
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