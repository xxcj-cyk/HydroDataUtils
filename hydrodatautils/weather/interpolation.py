'''
This code contains IDW interpolation method, spherical distance and plane distance calculation
Author: Yikai CHAI
'''

import numpy as np
from math import radians, cos, sin, asin, sqrt

def distance_lonlat(lon1, lat1, lon2, lat2):
    """
    Calculate spherical distance between two points

    Args:
        lon1 (float): Longitude of point 1
        lat1 (float): Latitude of point 1
        lon2 (float): Longitude of point 2
        lat2 (float): Latitude of point 2

    returns:
        d (float): Spherical distance between two points
    """
    R = 6372.8 # Set the constant for Earth's radius, unit is kilometers
    # Haversine Formula
    dLon = radians(lon2 - lon1)
    dLat = radians(lat2 - lat1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2
    c = 2 * asin(sqrt(a))
    d = R * c
    return d

def IDW(x, y, z, xi, yi):
    """
    Calculate the interpolation result for each point

    Args:
        x (array): x coordinates of known points
        y (array): y coordinates of known points
        z (array): values of known points
        xi (array): x coordinates of target points
        yi (array): y coordinates of target points

    returns:
        lstxyzi (list): x, y coordinates and interpolation results of target points
    """
    lstxyzi = [] # Used to store x, y coordinates and interpolation results of target points
    for p in range(len(xi)): # Iterate through x coordinates of target points
        for q in range(len(yi)): # Iterate through y coordinates of target points
            lstdist = [] # Used to store the distance from this target point to all known points
            for s in range(len(x)): # Iterate through known points
                d = distance_lonlat(x[s], y[s], xi[p], yi[q]) # Calculate the spherical distance from this target point to all known points
                lstdist.append(d)
            # Calculate interpolation
            '''
            np.power(base, exponent)
            Adjustable parameter: the exponent in parentheses can be input as 1 or 2
            '''
            w = list((1 / np.power(lstdist, 2))) # Weight
            sumw = np.sum(w) # Sum of weights
            sumwzi = np.sum(np.array(w) * np.array(z)) # Weighted sum
            u = sumwzi / sumw # Weighted average
            xyzi = [xi[p], yi[q], u] # Combine x, y coordinates and interpolation result of target point into a list
            lstxyzi.append(xyzi)
    return lstxyzi