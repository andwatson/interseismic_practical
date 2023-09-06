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
from cmcrameri import cm # this is additional scientific colour maps, see "https://www.fabiocrameri.ch/colourmaps/"
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

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

def shear_strain(x, s, d):
    '''
    Calculate shear strain rate based on slip rate and locking depth from a 
    screw dislocation model. From Savage & Burford (1973).
    
    INPUTS
        x = vector of distances from fault
        s = slip or slip rate on deep dislocation
        d = locking depth [same units as x]
    OUTPUTS
        e = shear strain [per year]
          
    USEAGE:
        e = shear_strain(x, s, d)
    '''
    
    e = ((s*d)/(2*np.pi)) * np.power((x.astype(float)**2 + d**2),-1)
    
    return e
    
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
    x = either a vector or array of coords for the x axis
    y = either a vector or array of coords for the y axis
    prof_start = (x, y) pair for the start of the profile line
    prof_end = (x, y) pair for the end of the profile line
    params = dictionary of parameters for the profiler (currently nbins and width)
    
    '''
    
    # If provided vector of coords, convert to arrays
    if x.ndim == 1 or y.ndim == 1:
        xx,yy = np.meshgrid(x,y)
    else:
        xx, yy = x, y
    
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

#-------------------------------------------------------------------------------

def sliding_window_mean(x, y, wind_size):
    '''
    A very simple sliding window average that crops the ends where the window
    would be undersize.
    
    INPUTS:
        x = distance
        y = value at distance
        wind_size = size of window, must be an odd integer
        
    OUTPUTS:
        x_smooth = reduced x coords
        y_smooth = mean for each window
    '''
    
    # make sure window size is an odd number
    assert (wind_size % 2) == 1, "Window size must be an odd number."
    
    # number of elements in output
    n_out = x.size-(wind_size-1)
    
    # size of window either side of centre point
    wind_half = int((wind_size-1)/2)
    
    # reduced x
    x_smooth = x[wind_half:-wind_half]
    
    # pre-allocate
    y_smooth = np.zeros(n_out)
    
    # loop through each window, taking the mean
    for ii in range(n_out):
        y_smooth[ii] = np.nanmean(y[ii:(ii+wind_size)])
    
    return x_smooth, y_smooth


def plot_screw_disc(x, s, d, v, s_ref, d_ref, v_ref, v_grid):
    # Let's break this first plotting section down

    # create a figure and axes for our plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # plot the model velocities as a line to the first axes/subplot
    axs[0].plot(x, v * 1000, color="blue", label='Your model: \ns = ' + str(s) + ' mm/yr, d = ' + str(d) + ' km')

    # plot the reference model
    axs[0].plot(x, v_ref * 1000, color="red", label='Reference model: \ns = {} mm/yr, d = {} km'.format(s_ref, d_ref))

    # draw dashed lines at x=0 (the fault trace) and at y=0
    axs[0].plot([0, 0], [axs[0].get_ylim()[0], axs[0].get_ylim()[1]], color='grey', linestyle='dashed')
    axs[0].plot([axs[0].get_xlim()[0], axs[0].get_xlim()[1]], [0, 0], color='grey', linestyle='dashed')

    # add a legend
    axs[0].legend(fontsize=12)

    # set labels for the x and y axes
    axs[0].set_xlabel('Fault-perpendicular distance (km)')
    axs[0].set_ylabel('Fault-parallel velocity (mm/yr)')

    # colour plot of our 2D velocities, now on our second axes/subplot
    im = axs[1].imshow(v_grid * 1000, extent=[x.min(), x.max(), x.min(), x.max()], cmap=cm.vik)

    # add a labeled colour bar
    plt.colorbar(im, ax=axs[1], label="Fault-parallel velocity (mm/yr)")

    # add the fault as a solid line
    axs[1].plot([x.min(), x.max()], [0, 0], color='black', label='Fault trace')

    # add the profile as a dashed line
    axs[1].plot([0, 0], [x.min(), x.max()], color='black', linestyle='dashed', label="Profile line")

    # add legend
    axs[1].legend(fontsize=12)

    # set labels for the x and y axes
    axs[1].set_xlabel('x-coord (km)')
    axs[1].set_ylabel('y-coord (km)')

    # this alters subplot spacing
    fig.tight_layout()

    # display the plot
    plt.show()


def plot_screw_disc_in_los(x, v, inc_grid, v_grid, v_grid_los):
    '''
    Plot a 2x2 panel of profiles extracted from incidence array, velocity array and los velocity array.
    Draw fault perpendicular profiles in near, mid and far range to show the effect of changing incidence angle along range
    '''
    # plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # get colour limits for grids
    cmin, cmax = np.amin(v_grid * 1000), np.amax(v_grid * 1000)

    # incidence angle grid
    im = axs[0, 0].imshow(inc_grid, extent=[x.min(), x.max(), x.min(), x.max()], cmap=cm.batlow)
    plt.colorbar(im, ax=axs[0, 0], label="Incidence angle (degrees)")
    axs[0, 0].plot([int(x.min()/2), int(x.min()/2)], [x.min(), x.max()], color='black', linestyle='dotted')
    axs[0, 0].plot([0, 0], [x.min(), x.max()], color='black', linestyle='dashed')
    axs[0, 0].plot([int(x.max()/2), int(x.max()/2)], [x.min(), x.max()], color='black', linestyle='solid')

    # East-west velocity grid
    im = axs[0, 1].imshow(v_grid * 1000, extent=[x.min(), x.max(), x.min(), x.max()], cmap=cm.vik, vmin=cmin, vmax=cmax)
    plt.colorbar(im, ax=axs[0, 1], label="East-west velocity (mm/yr)")
    axs[0, 1].plot([x.min(), x.max()], [0, 0], color='black', label='Fault trace')
    axs[0, 1].plot([0, 0], [x.min(), x.max()], color='red', linestyle='dashed', label="Profile line")
    axs[0, 1].legend(fontsize=12)

    # 1D models
    axs[1, 0].plot(x, v * 1000, color='red', label='original (East-West) velocity')
    axs[1, 0].plot(x, v_grid_los[:, 50] * 1000, color='black', linestyle='dotted', label='LOS velocity near range')
    axs[1, 0].plot(x, v_grid_los[:, 100] * 1000, color='black', linestyle='dashed', label='LOS velocity mid range')
    axs[1, 0].plot(x, v_grid_los[:, 150] * 1000, color='black', linestyle='solid', label='LOS velocity far range')

    axs[1, 0].plot([0, 0], [axs[1, 0].get_ylim()[0], axs[1, 0].get_ylim()[1]], color='grey', linestyle='dashed')
    axs[1, 0].plot([axs[1, 0].get_xlim()[0], axs[1, 0].get_xlim()[1]], [0, 0], color='grey', linestyle='dashed')
    axs[1, 0].set_xlabel('Fault-perpendicular distance (km)')
    axs[1, 0].set_ylabel('Velocity (mm/yr)')
    axs[1, 0].legend()

    # Line-of-sight velocity grid
    im = axs[1, 1].imshow(v_grid_los * 1000, extent=[x.min(), x.max(), x.min(), x.max()], cmap=cm.vik, vmin=cmin, vmax=cmax)
    plt.colorbar(im, ax=axs[1, 1], label="Line-of-sight velocity (mm/yr)")
    axs[1, 1].plot([x.min(), x.max()], [0, 0], color='black')
    axs[1, 1].plot([-100, -100], [x.min(), x.max()], color='black', linestyle='dotted')
    axs[1, 1].plot([0, 0], [x.min(), x.max()], color='black', linestyle='dashed')
    axs[1, 1].plot([100, 100], [x.min(), x.max()], color='black', linestyle='solid')

    plt.tight_layout()
    plt.show()


def plot_maps_with_reference(vel_asc_regrid, lon_regrid, lat_regrid, fault_trace, ref_poly, vel_desc_regrid):
    # plot the regridded velocities to make sure they're fine
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))

    # ascending velocities
    im = axs[0].imshow(vel_asc_regrid, extent=[lon_regrid[0], lon_regrid[-1], lat_regrid[0], lat_regrid[-1]], \
                       cmap=cm.vik, vmin=-20, vmax=20)
    axs[0].plot(fault_trace[:, 0], fault_trace[:, 1], color="red")
    axs[0].plot(ref_poly[:, 0], ref_poly[:, 1], color='black')
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad="2%")
    plt.colorbar(im, cax=cax, label='LOS velocity (mm/yr)')
    axs[0].set_title('087A_04904_121313')
    axs[0].set_xlim(np.amin(lon_regrid), np.amax(lon_regrid))
    axs[0].set_ylim(np.amin(lat_regrid), np.amax(lat_regrid))

    # descending velocities
    im = axs[1].imshow(vel_desc_regrid, extent=[lon_regrid[0], lon_regrid[-1], lat_regrid[0], lat_regrid[-1]], \
                       cmap=cm.vik, vmin=-20, vmax=20)
    axs[1].plot(fault_trace[:, 0], fault_trace[:, 1], color="red")
    axs[1].plot(ref_poly[:, 0], ref_poly[:, 1], color='black')
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad="2%")
    plt.colorbar(im, cax=cax, label='LOS velocity (mm/yr)')
    axs[1].set_title('167D_04884_131212')
    axs[1].set_xlim(np.amin(lon_regrid), np.amax(lon_regrid))
    axs[1].set_ylim(np.amin(lat_regrid), np.amax(lat_regrid))

    fig.tight_layout(w_pad=3)
    plt.show()


def plot_decomposed_maps(vel_para, vel_U,lon_regrid, lat_regrid, fault_trace, poly_asc, poly_desc):
    # plot the results
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))

    # East velocities
    im = axs[0].imshow(vel_para, extent=[lon_regrid[0], lon_regrid[-1], lat_regrid[0], lat_regrid[-1]], \
                       cmap=cm.vik, vmin=-20, vmax=20)
    axs[0].plot(fault_trace[:, 0], fault_trace[:, 1], color="red")
    axs[0].plot(poly_asc[:, 0], poly_asc[:, 1], color="black", linestyle='dashed')
    axs[0].plot(poly_desc[:, 0], poly_desc[:, 1], color="black", linestyle='dashed')
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad="2%")
    plt.colorbar(im, cax=cax, label='Fault-parallel velocity (mm/yr)')
    axs[0].set_title('Fault-parallel')
    axs[0].set_xlim(np.amin(lon_regrid), np.amax(lon_regrid))
    axs[0].set_ylim(np.amin(lat_regrid), np.amax(lat_regrid))

    # Up velocities
    im = axs[1].imshow(vel_U, extent=[lon_regrid[0], lon_regrid[-1], lat_regrid[0], lat_regrid[-1]], \
                       cmap=cm.vik, vmin=-20, vmax=20)
    axs[1].plot(fault_trace[:, 0], fault_trace[:, 1], color="red")
    axs[1].plot(poly_asc[:, 0], poly_asc[:, 1], color="black", linestyle='dashed')
    axs[1].plot(poly_desc[:, 0], poly_desc[:, 1], color="black", linestyle='dashed')
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad="2%")
    plt.colorbar(im, cax=cax, label='Vertical velocity (mm/yr)')
    axs[1].set_title('Up')
    axs[1].set_xlim(np.amin(lon_regrid), np.amax(lon_regrid))
    axs[1].set_ylim(np.amin(lat_regrid), np.amax(lat_regrid))

    plt.tight_layout(w_pad=3)
    plt.show()

def plot_utm_maps(xx_utm, yy_utm, vel_para, vel_U, fault_trace_utm, xlim, ylim):
    # replot with new coordinates
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # East velocities
    im = axs[0].scatter(xx_utm.flatten(), yy_utm.flatten(), s=2, c=vel_para.flatten(), cmap=cm.vik, vmin=-20, vmax=20)
    axs[0].plot(fault_trace_utm[:, 0], fault_trace_utm[:, 1], color="red")
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad="2%")
    plt.colorbar(im, cax=cax, label='Fault-parallel velocity (mm/yr)')
    axs[0].set_aspect('equal', 'box')
    axs[0].set_title('Fault-parallel')
    axs[0].set_xlabel('x-coord (km)')
    axs[0].set_ylabel('y-coord (km)')
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)

    # Up velocities
    im = axs[1].scatter(xx_utm.flatten(), yy_utm.flatten(), s=2, c=vel_U.flatten(), cmap=cm.vik, vmin=-20, vmax=20)
    axs[1].plot(fault_trace_utm[:, 0], fault_trace_utm[:, 1], color="red")
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad="2%")
    plt.colorbar(im, cax=cax, label='Vertical velocity (mm/yr)')
    axs[1].set_aspect('equal', 'box')
    axs[1].set_title('Up')
    axs[1].set_xlabel('x-coord (km)')
    axs[1].set_ylabel('y-coord (km)')
    axs[1].set_xlim(xlim)
    axs[1].set_ylim(ylim)

    fig.tight_layout(w_pad=5)
    plt.show()

def plot_profile(xx_utm, yy_utm, vel_para, fault_trace_utm, prof_start, prof_end, points_poly, xlim, ylim,
                 points_dist, points_val, prof_bin_mids, bin_val, intersect_dist, intersect_angle):
    '''
    Plotting a profile in UTM coordinates in map view and in cross section

    INPUTS:
        xx_utm, yy_utm, vel_para = 2D array
        fault_trace_utm, points_poly = 2 column x,y coordinates of fault and profile corner coordinates
        prof_start, prof_end = tuple x,y coordinates for end points of profile in map view
        points_poly = corner coordinates of profile rectangle
        xlim, ylim = to focus on area in the plot with real data
        points_dist, points_val = blue scatter points along profile
        prof_bin_mids, bin_val = red sliding mean along profile
        intersect_dist = to draw the grey dashed line to mark fault-profile intersection on the cross section
        intersect_angle = for text labelling
    '''
    # plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # East velocities
    im = axs[0].scatter(xx_utm.flatten(), yy_utm.flatten(), s=2, c=vel_para.flatten(), cmap=cm.vik, vmin=-20, vmax=20)
    axs[0].plot(fault_trace_utm[:, 0], fault_trace_utm[:, 1], color="red")
    axs[0].plot([prof_start[0], prof_end[0]], [prof_start[1], prof_end[1]], color="red")
    axs[0].plot(points_poly[:, 0], points_poly[:, 1], color="red")
    axs[0].scatter(prof_start[0], prof_start[1], s=100, color='pink', edgecolor='black', zorder=3)
    axs[0].scatter(prof_end[0], prof_end[1], s=100, color='black', edgecolor='white', zorder=3)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad="2%")
    plt.colorbar(im, cax=cax, label='Fault-parallel velocity (mm/yr)')
    axs[0].set_aspect('equal', 'box')
    axs[0].set_title('Fault-parallel')
    axs[0].set_xlabel('x-coord (km)')
    axs[0].set_ylabel('y-coord (km)')
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)

    # profile
    axs[1].scatter(points_dist, points_val)
    axs[1].plot(prof_bin_mids, bin_val, color="red")
    axs[1].plot([intersect_dist, intersect_dist], [axs[1].get_ylim()[0], axs[1].get_ylim()[1]], color='grey',
                linestyle='dashed')
    axs[1].plot([axs[1].get_xlim()[0], axs[1].get_xlim()[1]], [0, 0], color='grey', linestyle='dashed')
    axs[1].text(0.02, 0.90, 'intersection angle = ' + str(round(np.abs(intersect_angle))) + ' degrees', fontsize=14,
                transform=axs[1].transAxes)
    axs[1].set_xlabel("Distance along profile (km)")
    axs[1].set_ylabel("Fault-parallel velocity (mm/yr)")
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()

    fig.tight_layout(w_pad=4)
    plt.show()


def plot_creep_profile(x_prof, points_val, x, v, s1, s2, d1, d2, c, rms_forward):
    # Plot comparison
    fig, axs = plt.subplots(1, 1, figsize=(15, 8))

    plt.scatter(x_prof, points_val)
    plt.plot(x, v * 1000, c='r')
    plt.plot([0, 0], [axs.get_ylim()[0], axs.get_ylim()[1]], color='grey', linestyle='dashed')

    plt.text(0.02, 0.9, 'slip = ' + str(s1) + ' mm/yr', fontsize=14, transform=axs.transAxes)
    plt.text(0.25, 0.9, 'creep = ' + str(s2) + ' mm/yr', fontsize=14, transform=axs.transAxes)
    plt.text(0.02, 0.84, 'locking depth = ' + str(d1) + ' km', fontsize=14, transform=axs.transAxes)
    plt.text(0.25, 0.84, 'creep depth = ' + str(d2) + ' km', fontsize=14, transform=axs.transAxes)
    plt.text(0.02, 0.78, 'offset = ' + str(c) + ' mm/yr', fontsize=14, transform=axs.transAxes)
    plt.text(0.02, 0.65, 'RMS misfit = ' + str(round(rms_forward, 3)) + ' mm/yr', fontsize=14, transform=axs.transAxes)

    plt.xlabel('Perp. distance from fault (km)')
    plt.ylabel('Fault-parallel velocity (mm/yr)')

    plt.show()


def plot_profile_model(x_prof, points_val, x, v, s, d, c, rms_forward):
    # Plot comparison
    fig, axs = plt.subplots(1, 1, figsize=(15, 8))

    plt.scatter(x_prof, points_val)
    plt.plot(x, v * 1000, c='r')
    plt.plot([0, 0], [axs.get_ylim()[0], axs.get_ylim()[1]], color='grey', linestyle='dashed')
    plt.plot([axs.get_xlim()[0], axs.get_xlim()[1]], [0, 0], color='grey', linestyle='dashed')

    plt.text(0.15, 0.9, 'slip rate = ' + str(s) + ' mm/yr', fontsize=14, transform=axs.transAxes)
    plt.text(0.15, 0.84, 'locking depth = ' + str(d) + ' km', fontsize=14, transform=axs.transAxes)
    plt.text(0.15, 0.78, 'offset = ' + str(c) + ' mm/yr', fontsize=14, transform=axs.transAxes)
    plt.text(0.15, 0.72, 'RMS misfit = ' + str(round(rms_forward, 3)) + ' mm/yr', fontsize=14, transform=axs.transAxes)

    plt.xlabel('Perpendicular distance from surface fault trace (km)')
    plt.ylabel('Fault-parallel velocity (mm/yr)')

    plt.show()


def plot_strain_profile(x_prof, points_val, x, v, intersect_angle, e_shear, e_shear_grad):
    # Plot the original profile and model, and strain rate
    fig, axs = plt.subplots(2, 1, figsize=(15, 15))

    axs[0].scatter(x_prof, points_val)
    axs[0].plot(x, v * 1000, c='r')
    axs[0].plot([0, 0], [axs[0].get_ylim()[0], axs[0].get_ylim()[1]], color='grey', linestyle='dashed')
    axs[0].text(0.02, 0.90, 'intersection angle = ' + str(round(np.abs(intersect_angle))) + ' degrees', fontsize=14,
                transform=axs[0].transAxes)
    axs[0].set_xlabel("Distance from fault (km)")
    axs[0].set_ylabel("Fault-parallel velocity (mm/yr)")
    axs[0].yaxis.set_label_position("right")
    axs[0].yaxis.tick_right()

    axs[1].plot(x, e_shear, c='r', label='Forward model')
    axs[1].plot(x, e_shear_grad, c='b', label='Velocity gradient')
    axs[1].plot([0, 0], [axs[1].get_ylim()[0], axs[1].get_ylim()[1]], color='grey', linestyle='dashed')
    axs[1].legend(fontsize='x-large')
    axs[1].set_xlabel("Distance from fault (km)")
    axs[1].set_ylabel("Shear strain rate (/yr)")
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()

    plt.show()