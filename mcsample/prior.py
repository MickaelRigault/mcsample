#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Where generic priors are """

import numpy   as np
from scipy import stats

# ===================== #
#                       #
#     PRIORS            #
#                       #
# ===================== #
def normalflat_prior(value, loc, scale, boundaries=[None, None]):
    """
    Set up a flat prior with a normal one on a parameter.
    
    Parameters
    ----------
    value : [float]
        The paramater to apply the prior on.
    
    loc : [float]
        Mean of the normal prior.
    
    scale : [float]
        Scale of the normal prior.
    
    boundaries : [list[float]]
        Boundaries of the flat prior.
    
    
    Returns
    -------
    float
    """
    return normal_prior(value, loc, scale)*flat_prior(value, boundaries=boundaries)
    
def normal_prior(value, loc, scale):
    """
    Set up a normal prior on a parameter.
    
    Parameters
    ----------
    value : [float]
        The paramater to apply the prior on.
    
    loc : [float]
        Mean of the normal prior.
    
    scale : [float]
        Scale of the normal prior.
    
    
    Returns
    -------
    float
    """
    return stats.norm.pdf(value, loc=loc, scale=scale) 

def flat_prior(value, boundaries=[-np.inf, +np.inf]):
    """
    Set up a flat prior on a paramater.
    Out of the bounds, the prior return 0, else it returns 1.
    
    Parameters
    ----------
    value : [float]
        The paramater to apply the prior on.
    
    boundaries : [list[float]]
        Boundaries of the flat prior.
    
    
    Returns
    -------
    float
    """
    return stats.uniform.pdf(value, *boundaries)
    
