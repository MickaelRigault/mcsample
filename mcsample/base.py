#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Basic library tools """

import pandas
import numpy as np
from scipy import stats
# Markov Chain Monte Carlo
import emcee
# MR libs
from propobject import BaseObject


class MCMCHandler( BaseObject ):
    """ """
    PROPERTIES = ["sampler", "walkers", "nsteps", "warmup"]
    SIDE_PROPERTIES = ["nchains"]
    
    def __init__(self, sampler):
        """ """
        self.set_sampler(sampler)

    # ------- #
    #  Main   #
    # ------- #
    def run(self, guess=None, nchains=None, nsteps=2000, warmup=500):
        """ """
        self.set_steps(nsteps, warmup)
        pos, prob, state = self.walkers.run_mcmc(self.get_guesses(guess), self._total_steps)
        
    # ------- #
    # SETTER  #
    # ------- #
    def set_sampler(self, sampler):
        """ """
        if Sampler not in sampler.__class__.__mro__:
            raise TypeError("given sampler is not a Sampler object (nor inherite from)")
        self._properties["sampler"] = sampler

    def set_steps(self, nsteps, warmup):
        """ """
        self._properties["nsteps"] = int(nsteps)
        self._properties["warmup"] = int(warmup)

    def adjust_warmup(self, warmup):
        """ change the relative warmup to steps ratio """
        if self._properties["nsteps"] is None:
            raise AttributeError("steps and warmup not defined yet, please run set_steps()")
        warmup = int(warmup)
        self.set_steps(self._total_steps - warmup, warmup)
        
    def set_nchains(self, nchains):
        """ """
        self._side_properties["nchains"] = nchains
        
    def setup(self, nchains=None):
        """ """
        if nchains is not None:
            self.set_nchains(nchains)
            
        self._properties["walkers"] = emcee.EnsembleSampler(self.nchains, self.nfreeparameters, self.sampler.get_logprob)

    # ------- #
    # GETTER  #
    # ------- #
    def get_guesses(self, guess=None):
        """ """
        
        guess = np.zeros(self.nfreeparameters) if guess is None else np.asarray(guess)
        return (guess[:, np.newaxis] * (1+1e-2*np.random.randn(self.nchains))).T
    
    # =================== #
    #   Parameters        #
    # =================== #
    @property
    def sampler(self):
        """ """
        return self._properties["sampler"]

    @property
    def walkers(self):
        """ """
        return self._properties["walkers"]

    @property
    def chains(self):
        """ """
        return self.walkers.chain[:, self.warmup:, :].reshape((-1, self.nfreeparameters)).T

    @property
    def _chains_full(self):
        """ """
        return self.walkers.chain.reshape((-1, self.nfreeparameters)).T

    @property
    def derived_values(self):
        """ 3 times N array of the derived parameters
            [50%, +1sigma (to 84%), -1sigma (to 16%)]
        """
        return map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(self.chains, [16, 50, 84],axis=0)))
    
    @property
    def derived_parameters(self):
        """ dictionary of the mcmc derived values with the structure:
           NAME_OF_THE_PARAMETER = 50% pdf
           NAME_OF_THE_PARAMETER.err = [+1sigma, -1sigma]
        """
        fitout = {}
        for v,name in zip(self.derived_values, self.freeparameters):
            fitout[name] = v[0]
            fitout[name+".err"] = [v[1],v[2]]
            
        return fitout
    
    # Number of steps
    @property
    def nsteps(self):
        """ number of steps post warmup"""
        return self._properties["nsteps"]

    @property
    def warmup(self):
        """ number of warmup steps """
        return self._properties["warmup"]

    @property
    def _total_steps(self):
        """ """
        return self.nsteps + self.warmup
    
    @property
    def nchains(self):
        """ number of chains. 2 times the number of free parameters by default """
        if self._side_properties["nchains"] is None:
            return self.nfreeparameters * 2
        return self._side_properties["nchains"]
    
    # From Sampler    
    @property
    def freeparameters(self):
        """ short cut to self.sampler.freeparameters """
        return self.sampler.freeparameters
    
    @property
    def nfreeparameters(self):
        """ short cut to self.sampler.freeparameters """
        return self.sampler.nfreeparameters
    

class Sampler( BaseObject ):
    """
    This class makes the MCMC sampler using the library "emcee".
    """
    
    PROPERTIES         = ["data", "parameters", "freeparameters", "nb_chains", "mcmc"]
    SIDE_PROPERTIES    = []
    DERIVED_PROPERTIES = []
    PARAMETERS         = None
    
    def __init__(self, data=None, **kwargs):
        """
        Initialization.
        Can execute set_data().
        
        Parameters
        ----------
        data : [dict or pandas.DataFrame]
            Dataset.
        
        
        Returns
        -------
        Void
        """
        if data is not None:
            self.set_data(data, **kwargs)
        
    # ------- #
    # SETTER  #
    # ------- #
    def set_parameters(self, param, index=None):
        """
        Associate the fitted parameter names to their value.
        If index is None, every fitted parameter is settled.
        If not, only the index ones are.
        
        Parameters
        ----------
        param : [list[float] or None]
            List of fitted parameter values.
        
        index : [list[string] or None]
            List of "param" input associated fitted parameter names.
        
        
        Returns
        -------
        Void
        """
        if index is None:
            self._properties["parameters"] = {k:v for k,v in zip(self.freeparameters, param)}
        else:
            if self._properties["parameters"] == None:
                self._properties["parameters"] = {}
            for ii, ii_index in enumerate(index if type(index)==list else [index]):
                self._properties["parameters"][ii_index] = (param if type(param)==list else [param])[ii]
        
    def set_data(self, data):
        """
        Convert an input data dictionnary (or DataFrame) into a DataFrame to use in MCMC.
        
        Parameters
        ----------
        data : [dict or pandas.DataFrame]
            Dataset, it excpects to contain hubble residuals as 'hr' and 'hr.err' and the age tracer reference data.
        
        
        Returns
        -------
        Void
        """
        if type(data) is pandas.DataFrame:
            self._properties["data"] = data
        elif type(data) is dict:
            self._properties["data"] = pandas.DataFrame(data)
        else:
            raise TypeError("data must be a DataFrame or a dict")

    def define_free_parameters(self, freeparameters):
        """
        Define the parameter names to fit by the MCMC sampler.
        
        Parameters
        ----------
        freeparameters : [string or list[string]]
            List of the names of the parameters to fit.
        
        
        Returns
        -------
        Void
        """
        freeparameters = freeparameters if type(freeparameters)==list else [freeparameters]
        self._properties["freeparameters"] = freeparameters
        
    # - POSTERIOR
    def get_logprob(self, param=None):
        """
        Combine the values from get_logprior and get_loglikelihood to set the log probability which will be maximized by the MCMC sampler.
        
        Parameters
        ----------
        param : [list[float] or None]
            List of fitted parameter values.
        
        
        Returns
        -------
        float
        """
        if param is not None:
            self.set_parameters(param)
        
        # Tested necessary to avoid NaN and so
        log_prior = self.get_logprior()
        if not np.isfinite(log_prior):
            return -np.inf
            
        
        return log_prior + self.get_loglikelihood() 
        
    #
    # Overwrite
    #   
    # - PRIOR 
    def get_logprior(self, param=None):
        """
        Return the sum of the log of the prior values returned for every concerned parameter.
        Each one fall within the interval [-inf, 0].
        
        Parameters
        ----------
        param : [list[float] or None]
            List of fitted parameter values.
        
        
        Returns
        -------
        float
        """
        # - Reminder
        #
        # Code: To add a prior, add a variable called prior_BLA = TOTOTO
        #
        priors_ = np.asarray(self.get_prior_list(param=param))
        
        return np.sum(np.log(priors_)) if np.all(priors_>0) else -np.inf

    def get_prior_list(self, param=None):
        """
        Call the so called function in the child class.
        
        Parameters
        ----------
        param : [list[float] or None]
            List of fitted parameter values.
        """
        if param is not None:
            self.set_parameters(param)
            
        raise NotImplementedError("You must define get_prior_list() ")
          
    # - LIKELIHOOD
    def get_loglikelihood(self, param=None):
        """
        Call the so called function in the child class.
        
        Parameters
        ----------
        param : [list[float] or None]
            List of fitted parameter values.
        """
        if param is not None:
            self.set_parameters(param)
            
        raise NotImplementedError("You must define get_loglikelihood() ")

    # =========== #
    #  emcee      #
    # =========== #     
    def run_mcmc(self, guess=None, nchains=None, warmup=1000, nsteps=2000, verbose=True):
        """
        Run the emcee sampling.
        First step is the warmup, from which the result is used to initialize the true sampling.
        
        Parameters
        ----------
        guess : [None or list[float]]
            List of the initial guess for each fitted parameter.
        
        nchains : [int or None]
            Number of chains to run the whole MCMC sampling.
            Minimum, and the default value, is two times the number of fitted parameters.

        warmup : [int]
            Number of iterations to run the warmup step.
        
        nsteps : [int]
            Number of iterations to run the true sampling.
        
        
        Options
        -------
        verbose : [bool]
            Option to show MCMC progress and the time taken to run.
        
        
        Returns
        -------
        Void
        """
        self.mcmc.run(guess, nsteps=nsteps, warmup=warmup, nchains=nchains)
        
    # ================ #
    #  Properties      #
    # ================ #
    @property
    def freeparameters(self):
        """ list of fitted parameter names """ 
        if self._properties["freeparameters"] is None and self.PARAMETERS is not None:
            self._properties["freeparameters"] = self.PARAMETERS
        return self._properties["freeparameters"]
        
    @property
    def parameters(self):
        """ dictionnary of each fitted parameter """
        return self._properties["parameters"]
    
    @property
    def nfreeparameters(self):
        """ number of fitted parameters """
        return len(self.freeparameters)

    @property
    def chains(self):
        """ mcmc chains flatten (after warmup) """
        return self.mcmc.chains
        
    @property
    def data(self):
        """ pandas DataFrame containing the data """
        return self._properties["data"]

    @property
    def mcmc(self):
        """ """
        if self._properties["mcmc"] is None:
            self._properties["mcmc"] = MCMCHandler(self)
        return self._properties["mcmc"]













