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


__all__ = ["chain_to_median_error", "Sampler"]

def chain_to_median_error(chain, structure=[16,50,84]):
    """
    Returns [value, -error, +error], defined by the given structure.
    
    Parameters
    ----------
    chain : [np.array]
        Array containing the MCMC chain on which to recover the median and the errors.
    
    structure : [list]
        Percentiles to get from the given chain.
        The structure must be [-error, value, +error] percentiles.
        Default is [16,50,84].
    
    
    Returns
    -------
    np.array
    """
    if len(np.shape(chain)) == 1:
        v = np.percentile(chain, structure, axis=0)
        return np.asarray((v[1], v[1]-v[0], v[2]-v[1]))
    
    return np.asarray([(v[1], v[1]-v[0], v[2]-v[1]) for v in np.percentile(chain, structure, axis=1).T])


class MCMCHandler( BaseObject ):
    """
    "emcee" MCMC handler.
    """
    
    PROPERTIES = ["sampler", "walkers", "nsteps", "warmup"]
    SIDE_PROPERTIES = ["nchains"]
    DERIVED_PROPERTIES = ["pltcorner"]
    
    def __init__(self, sampler):
        """
        Initialization, run 'set_sampler'.
        
        Parameters
        ----------
        sampler : [Sampler]
            Sampler object (cf. Samper class).
        
        
        Returns
        -------
        Void
        """
        self.set_sampler(sampler)

    # ------- #
    #  Main   #
    # ------- #
    def run(self, guess=None, nchains=None, nsteps=2000, warmup=500, kwargs=None, verbose=True):
        """
        Run "emcee" sampler.
        
        Parameters
        ----------
        guess : [list or np.array or None]
            List of guess for each free parameter.
            If None, the default guess for every free parameter.
            Default is None.
        
        nchains : [int or None]
            Number of chains for the sampling. It must be more than or equal to two times the number of free parameters.
            If None, the number of chains is two times the number of free parameters.
            Default is None.
        
        nsteps : [int]
            Number of steps for sampling.
            Default if 2000.
        
        warmup : [int]
            Number of steps for warmup.
            Default is 500.
        
        kwargs : [dict]
            Additional paramters on the likelihood (self.sampler.get_logprob).
            Default is None.
        
        Options
        -------
        verbose : [bool]
            If True, print sampling states.
            Default is True.
        
        
        Returns
        -------
        Void
        """
        self.set_steps(nsteps, warmup)
        self.setup(nchains=nchains, kwargs=kwargs)
        
        if verbose:
            from time import time
            t0 = time()
            for ii, (pos, prob, state) in enumerate(self.walkers.sample(self.get_guesses(guess), iterations=self._total_steps)):
                self._verbose_mcmc_printer_(ii)
            # How long?
            t1 = time()
            time_mcmc = t1-t0
            print("Time taken to run 'emcee' is {0:.3f} seconds".format(time_mcmc))
        else:
            pos, prob, state = self.walkers.run_mcmc(self.get_guesses(guess), self._total_steps)

    def _verbose_mcmc_printer_(self, ii):
        """
        Print MCMC sampling state.
        
        Parameters
        ----------
        ii : [int]
            Sampling iteration.
        
        
        Returns
        -------
        Void
        """
        percentage = ii*self.nchains*100./(self._total_steps*self.nchains)
        if ii <= self.warmup and percentage % 10 == 0:
            print("{0}/{1} --> {2:.1f}% : Warmup".format(ii*self.nchains, (self._total_steps*self.nchains), percentage))
        elif ii == self.warmup or ii > self.warmup and percentage % 10 == 0:
            print("{0}/{1} --> {2:.1f}% : Sampling".format(ii*self.nchains, (self._total_steps*self.nchains), percentage))
        elif ii == self._total_steps - 1:
            print("{0}/{1} --> {2:.1f}% : Sampling".format((self._total_steps*self.nchains), (self._total_steps*self.nchains), 100.))
        
    # ------- #
    # SETTER  #
    # ------- #
    def set_sampler(self, sampler):
        """
        Set the sampler as an attribute.
        
        Parameters
        ----------
        sampler : [Sampler]
            Sampler object (cf. Sampler class).
        
        
        Returns
        -------
        Void
        """
        if Sampler not in sampler.__class__.__mro__:
            raise TypeError("given sampler is not a Sampler object (nor inherite from)")
        self._properties["sampler"] = sampler

    def set_steps(self, nsteps, warmup):
        """
        Set the chozen number of steps (sampling and warmup) as attributes.
        
        Parameters
        ----------
        nsteps : [int]
            Number of steps for sampling.
        
        warmup : [int]
            Number of steps for warmup.
        
        
        Returns
        -------
        Void
        """
        self._properties["nsteps"] = int(nsteps)
        self._properties["warmup"] = int(warmup)

    def adjust_warmup(self, warmup):
        """
        Change the relative warmup to steps ratio.
        
        Parameters
        ----------
        warmup : [int]
            Number of steps for the warmup.
        
        
        Returns
        -------
        Void
        """
        if self._properties["nsteps"] is None:
            raise AttributeError("steps and warmup not defined yet, please run set_steps(nsteps, warmup)")
        warmup = int(warmup)
        self.set_steps(self._total_steps - warmup, warmup)
        
    def set_nchains(self, nchains=None):
        """
        Set the number of chains as an attribute.
        
        Parameters
        ----------
        nchains : [int or None]
            Number of chains for the sampling. It must be more than or equal to two times the number of free parameters.
            If None, the number of chains is two times the number of free parameters.
            Default is None.
        
        
        Returns
        -------
        Void
        """
        self._side_properties["nchains"] = nchains
        

    def setup(self, nchains=None, kwargs=None):
        """
        Create a "emcee" sampler and set it as an attribute.
        
        Parameters
        ----------
        nchains : [int or None]
            Number of chains for the sampling. It must be more than or equal to two times the number of free parameters.
            If None, the number of chains is two times the number of free parameters.
            Default is None.
        
        kwargs : [dict]
            Additional paramters on the likelihood (self.sampler.get_logprob).
            Default is None.
        
        
        Returns
        -------
        Void
        """
        if nchains is not None:
            self.set_nchains(nchains)
            
        self._properties["walkers"] = emcee.EnsembleSampler(nwalkers=self.nchains, ndim=self.nfreeparameters,
                                                            log_prob_fn=self.sampler.get_logprob, kwargs=kwargs)

    # ------- #
    # GETTER  #
    # ------- #
    def get_guesses(self, guess=None):
        """
        Return an array containing the emcee compatible guesses.
        
        Parameters
        ----------
        guess : [list or np.array or None]
            List of guess for each free parameter.
            If None, the default guess for every free parameter.
            Default is None.
        
        
        Returns
        -------
        np.array
        """
        guess = np.zeros(self.nfreeparameters) if guess is None else np.asarray(guess)
        if guess.ndim == 1:
            return np.asarray([g* (1+1e-2*np.random.randn(self.nchains)) for g in guess]).T
        elif guess.ndim == 2:
            if guess.shape == (self.nchains, self.nfreeparameters):
                return guess
            else:
                raise ValueError("The shape of 'guess' must be (nb of walkers, nb of free paramaters).")
        else:
            raise ValueError(f"You gave a non compatible 'guess' argument:\n {guess}")

    # ------- #
    # PLOTTER #
    # ------- #
    def show(self, **kwargs):
        """
        Corner plot of the free parameters.
        
        **kwargs
        
        
        Returns
        -------
        Void
        """
        from .plot import MCCorner
        self._derived_properties["pltcorner"] = MCCorner(self)
        self.pltcorner.show(**kwargs)
        
    # =================== #
    #   Parameters        #
    # =================== #
    @property
    def sampler(self):
        """ Sampler object """
        return self._properties["sampler"]

    @property
    def walkers(self):
        """ walker arrays """
        return self._properties["walkers"]

    @property
    def chains(self):
        """ chain arrays without warmup steps """
        return self.walkers.chain[:, self.warmup:, :].reshape((-1, self.nfreeparameters)).T

    @property
    def _chains_full(self):
        """ full chain arrays (warmup + sampling) """
        return self.walkers.chain.reshape((-1, self.nfreeparameters)).T

    @property
    def pltcorner(self):
        """ MCCorner Plotting method (loaded during self.show()) """
        return self._derived_properties["pltcorner"]
        
    @property
    def derived_values(self):
        """ 3 times N array of the derived parameters [value, -error, +error] """
        return chain_to_median_error(self.chains)
    
    @property
    def derived_parameters(self):
        """ dictionary of the mcmc derived values with the structure:
           NAME_OF_THE_PARAMETER = 50% pdf
           NAME_OF_THE_PARAMETER.err = [-1sigma, +1sigma]
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
        """ total number of steps (warmup + sampling) """
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
        freeparameters : [string or list[string] or None]
            List of the names of the parameters to fit.
        
        
        Returns
        -------
        Void
        """
        freeparameters = freeparameters if (type(freeparameters)==list or freeparameters is None) else [freeparameters]
        self._properties["freeparameters"] = freeparameters
        
    # - POSTERIOR
    def get_logprob(self, param=None, **kwargs):
        """
        Combine the values from get_logprior and get_loglikelihood to set the log probability which will be maximized by the MCMC sampler.
        
        Parameters
        ----------
        param : [list[float] or None]
            List of fitted parameter values.
        
        **kwargs
        
        
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
            
        
        return log_prior + self.get_loglikelihood(**kwargs)
        
    #
    # Overwrite
    #   
    # - PRIOR 
    def get_logprior(self, param=None, verbose=False):
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
        
        
        Returns
        -------
        list
        """
        if param is not None:
            self.set_parameters(param)
            
        raise NotImplementedError("You must define get_prior_list() ")
          
    # - LIKELIHOOD
    def get_loglikelihood(self, param=None, **kwargs):
        """
        Call the so called function in the child class.
        
        Parameters
        ----------
        param : [list[float] or None]
            List of fitted parameter values.
        
        **kwargs
        
        
        Returns
        -------
        Void
        """
        if param is not None:
            self.set_parameters(param)
            
        raise NotImplementedError("You must define get_loglikelihood() ")

    # =========== #
    #  emcee      #
    # =========== #     
    def run_mcmc(self, guess=None, nchains=None, warmup=1000, nsteps=2000, verbose=True, kwargs=None):
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
        
        kwargs : [dict]
            Additional parameters.
        
        
        Returns
        -------
        Void
        """
        self.mcmc.run(guess, nsteps=nsteps, warmup=warmup, nchains=nchains, verbose=verbose, kwargs=kwargs)

        
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
        """ MCMCHandler object """
        if self._properties["mcmc"] is None:
            self._properties["mcmc"] = MCMCHandler(self)
        return self._properties["mcmc"]













