#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as mpl
from scipy.stats import gaussian_kde

from propobject import BaseObject

def _build_plotparam_(k_, nchains, paramname):
    """ """
    # - Axes
    k_ = np.atleast_1d(k_)
    if len(k_) == 1:
        k_ = [k_[0] for i in range(nchains)]
    elif len(k_) != nchains:
        raise ValueError("%s and chains must have the same size (%d vs. %d) "(paramname,len(k_),nchains))
    return k_


def plot_walkers(chains, axes=None, labels=None, colors=None, lws=1, alphas=1, lss="-", orientation="vertical"):
    """ """
    if orientation not in ["vertical", "horizontal"]:
        raise ValueError("orientation must be 'vertical' or 'horizontal', %s given"%orientation)
    
    nchains = len(chains)
    # - Axes
    if axes is None:
        fig = mpl.figure(figsize=[6,2*nchains])
        axes = [fig.add_subplot(nchains, 1, i+1) for i in range(nchains)]
    elif len(axes) != nchains:
        raise ValueError("axes and chains must have the same size (%d vs. %d) "(len(axes),nchains))
    else:
        fig = axes[0].figure
        
    colors = _build_plotparam_(colors, nchains, "colors")
    alphas = _build_plotparam_(alphas, nchains, "alphas")
    lss    = _build_plotparam_(lss,    nchains, "lss")
    lws    = _build_plotparam_(lws,    nchains, "lws")
    labels = _build_plotparam_(labels, nchains, "labels")
    for i in range(nchains):
        
        if orientation == "horizontal":
            axes[i].plot(chains[i], color=colors[i], alpha=alphas[i], 
                        lw=lws[i], ls=lss[i])
        else:
            xx_ = np.arange( len(chains[i]) )
            axes[i].plot(chains[i],xx_, color=colors[i], alpha=alphas[i], 
                        lw=lws[i], ls=lss[i])
            
        if labels[i] is not None:
            axes[i].set_ylabel(labels[i])
    return axes


def get_kde_contours(samples_i, samples_j, contours = [0.317311, 0.0455003], 
                 bw_method=0.5):
    """ 
    contours: [array]
        [0.317311, 0.0455003] means 1 and 2 sigma
    """
    #samples_i,samples_j = test_chains[0],test_chains[1]
    steps = 100
    xvals, yvals = np.meshgrid(np.linspace(min(samples_i), max(samples_i), steps),
                               np.linspace(min(samples_j), max(samples_j), steps))

    kernel = gaussian_kde(np.array([samples_i, samples_j]), bw_method = bw_method)

    eval_points = np.array([xvals.reshape(steps**2), yvals.reshape(steps**2)])

    kernel_eval = kernel(eval_points)
    norm_term = kernel_eval.sum()
    kernel_eval /= norm_term
    kernel_sort = np.sort(kernel_eval)
    kernel_eval = np.reshape(kernel_eval, (steps, steps))
    kernel_cum = np.cumsum(kernel_sort)

    levels = [kernel_sort[np.argmin(abs(kernel_cum - item))] for item in contours[::-1]]
    return  xvals, yvals, kernel_eval, levels



class MCCorner( BaseObject ):
    PROPERTIES = ["axes", "fig", "mcmchandler"]
    SIDE_PROPERTIES = ["contours", "contour_colors"]
    DERIVED_PROPERTIES = ["kdes"]
    
    def __init__(self, mcmchandler):
        """ """
        if mcmchandler is not None:
            self.set_mcmchandler(mcmchandler)

    # -------- #
    #  SETTER  #
    # -------- #
    def set_mcmchandler(self, mcmchandler):
        """ """
        self._properties["mcmchandler"] = mcmchandler
        
    def set_contours(self, contours, contour_colors=None, cmap="Blues"):
        """ 
        Parameters:
        ----------
        contours: [array]
            [0.317311, 0.0455003] means 1 and 2 sigma
        """
        if contour_colors is not None:
            if len(contour_colors) != len(contours):
                raise ValueError("contours and contour_colors must have the same size")
            self._side_properties["contour_colors"] = contour_colors
        else:
            self._side_properties["contour_colors"] = [mpl.cm.get_cmap(cmap)(0.1+ i/len(contours)*0.9) for i in range(len( contours))]

        self._side_properties["contours"] = contours
        
    def setup(self, left=0.1, bottom=0.1, right=0.1, top=0.1, span=None):
        """ """
        if span is None:
            span = np.max([0.015-0.002*np.sqrt(self.nfreeparameters), 0.005])
            
        self._properties["fig"] = mpl.figure( figsize=np.asarray([1,1])*(2+np.sqrt(self.nfreeparameters))*1.5 )
        width  = (1-(right+left))/self.nfreeparameters - span
        heigth = (1-(top+bottom))/self.nfreeparameters - span
    
        self._properties["axes"] = [[self.fig.add_axes([left+xone*(width+span), bottom+yone*(heigth+span), width, heigth]) 
                                        for xone in range(self.nfreeparameters)]
                                        for yone in range(self.nfreeparameters)][::-1]
    # -------- #
    #  LOADER  #
    # -------- #
    def load_kde(self, i, j, bw_method="scott"):
        """ """
        
        self.kdes[i][j] = get_kde_contours(self.mcmchandler.chains[j],
                                           self.mcmchandler.chains[i], #switch on purpose
                                           contours=self.contours,
                                           bw_method=bw_method)
        
    # --------- #
    #  PLOTTER  #
    # --------- #
    def show(self, newfig=True, labels=None,
                 show_walkers=True, 
                 worientation="vertical"):
        """ show the corner plot
        
        Parameters
        ----------
        newfig: [bool] -optional-
            Should this create a new fig (running self.setup())

        labels: [string/None] -optional-
            list of labels corresponding to the freeparameters
            - None means 'use the freeparameters names'
            - 'None' means no labeling
        
        show_walkers: [bool] -optional-
            Do you want to display the walker behind the diagonal 1d pdf ?
            
        worientation [string] -optional-
            orientiation of the walkers 
            worientation could be 'vertical' or 'horizontal'
            [ignored if show_walkers=False]

        Returns
        -------
        Void
        """
        if self.fig is None or newfig:
            self.setup()

        #return axes
        for xone in range(self.nfreeparameters):
            x = self.mcmchandler.chains[xone]
            for yone in range(self.nfreeparameters):
                y = self.mcmchandler.chains[yone]
                ax = self.axes[xone][yone]
                if xone==yone:
                    ax_hist = ax
                    xkde = np.linspace(*np.percentile(y,[0,100]),1000)
                    ykde = gaussian_kde(y, bw_method="scott")(xkde)
                    ax.plot(xkde, ykde, color=self.contour_colors[-1], lw=2)
                elif xone>yone:
                    self.display_contours(xone,yone)
                else:
                    ax.set_visible(False)

        self.cleanup_axes()
        if labels not in ["no_labels", "None"]:
            self.set_labels(labels)
        
        if show_walkers:
            self.show_walkers(orientation=worientation)

    
    def display_contours(self, i, j, **kwargds):
        """ """
        if self.kdes[i][j] is None:
            self.load_kde(i,j)

        # - Data
        xvals, yvals, kernel_eval, levels = self.kdes[i][j]
        # - Plot        
        prop = dict(levels = levels+[1], colors = self.contour_colors)
        self.axes[i][j].contour(xvals, yvals, kernel_eval, **{**prop,**kwargds})
        self.axes[i][j].contourf(xvals, yvals, kernel_eval, **{**prop,**kwargds})


    def show_walkers(self, axes=None, colors=None, orientation="vertical", **kwargs):
        """ """
        axes_w = [self.axes[i][i].twinx().twiny() for i in range(self.nfreeparameters)] if axes is None else axes
        plot_walkers( self.mcmchandler.chains,
                      axes=axes_w, colors=[self.contour_colors[-1]] if colors is None else colors,
                      orientation=orientation,
                      **{**dict(alphas=0.1, lws=0.5),**kwargs})
        if orientation == "horizontal":
            [axes_w[i].set_ylim(*np.asarray(self.axes).T[0][i].get_ylim()) for i in range(1,self.nfreeparameters)]
            axes_w[0].set_ylim(*np.asarray(self.axes).T[0][0].get_xlim())
        else:
            [axes_w[i].set_xlim(*self.axes[-1][i].get_xlim()) for i in range(self.nfreeparameters-1)]
            axes_w[-1].set_xlim(*self.axes[-1][0].get_ylim())
            
        [[ax_.set_xticks([]),ax_.set_yticks([])] for ax_ in axes_w]
        
    # - Cleaning Axes
    def cleanup_axes(self):
        """ """
        self.setup_axes_limits()
        self.setup_axes_ticks()

    def setup_axes_limits(self):
        """ """
        # Force shared xlim, ylim
        for i in range( len(self.axes)):
            [self.axes[j][i].set_xlim(*self.axes[i][i].get_xlim()) for j in range( len(self.axes)) if j>i]
            [self.axes[i][j].set_ylim(*self.axes[i][i].get_xlim()) for j in range( len(self.axes)) if j<i]

    def set_labels(self, labels=None, **kwargs):
        """ """
        if labels is None:
            labels = self.mcmchandler.freeparameters
        labels = _build_plotparam_(labels, self.nfreeparameters, "labels")
        [ax_.set_xlabel(label_, **kwargs) for ax_, label_ in zip(self.axes[-1], labels)]
        [ax_.set_ylabel(label_, **kwargs) for ax_, label_ in zip(np.asarray(self.axes).T[0][1:], labels)]
        
    def setup_axes_ticks(self):
        """ """
        # Labeling
        [self.axes[i][i].set_yticks([]) for i in range(self.nfreeparameters)]
        [[ax_.set_yticklabels(["" for i in ax_.get_yticklabels()]) for ax_ in axc] for axc in np.asarray(self.axes).T[1:]]
        [[ax_.set_xticklabels(["" for i in ax_.get_xticklabels()]) for ax_ in axc] for axc in self.axes[:-1]]
        
    # ================= #
    #   Properties      #
    # ================= #
    @property
    def mcmchandler(self):
        """ """
        return self._properties["mcmchandler"]
    
    @property
    def sampler(self):
        """ """
        return self.mcmchandler.sampler
    
    @property
    def nfreeparameters(self):
        """ """
        return self.mcmchandler.nfreeparameters

    # - Axes and Fig
    @property
    def axes(self):
        """ """
        return self._properties["axes"]
    
    @property
    def fig(self):
        """ """
        return self._properties["fig"]

    # - inside
    @property
    def contours(self):
        """ """
        if self._side_properties["contours"] is None:
            self.set_contours([0.317311, 0.0455003])
            
        return self._side_properties["contours"]

    @property
    def contour_colors(self):
        """ """
        if self._side_properties["contour_colors"] is None:
            self.set_contours([0.317311, 0.0455003])
        return self._side_properties["contour_colors"]
    
    @property
    def kdes(self):
        """ """
        if self._derived_properties["kdes"] is None:
            self._derived_properties["kdes"] = [[None for i in range(self.nfreeparameters)] for j in range(self.nfreeparameters)]
        return self._derived_properties["kdes"]
        




    


contours = [0.317311, 0.0455003]
def show_mcmc(self, walkers, labels=None):
    """ """
    from scipy.stats import gaussian_kde

    nwalk = np.size(walkers,axis=0)
    if labels is None:
        labels = ["param-%d"%i for i in range(nwalk)]
        
    fig    = mpl.figure(figsize=np.asarray([(nwalk+1),nwalk])*[8,8]/nwalk)
    left, bottom, right, top, span = 0.1,0.1,0.2,0.1, 0.008
    width  = (1-(right+left))/nwalk - span*2
    heigth = (1-(top+bottom))/nwalk - span*2
    
    axes = [[fig.add_axes([left+xone*(width+span), bottom+yone*(heigth+span), width, heigth]) 
            for xone in range(nwalk)]
            for yone in range(nwalk)][::-1]
    #return axes
    for xone in range(nwalk):
        x = walkers[xone]
        for yone in range(nwalk):
            y = walkers[yone]
            ax = axes[xone][yone]
            if xone==yone:
                ax_hist = ax
                xkde = np.linspace(*np.percentile(y,[0,100]),1000)
                ykde = gaussian_kde(y, bw_method="scott")(xkde)
                ax.plot(xkde, ykde, color=mpl.cm.Blues(0.55), lw=2)
            elif xone>yone:
                #ax.scatter(x,y, s=1)
                display_contours(ax, y,x, bw_method="scott", contours=contours)
            else:
                ax.set_visible(False)
    # Force shared xlim, ylim
    for i in range(len(axes)):
        [axes[j][i].set_xlim(*axes[i][i].get_xlim()) for j in range(len(axes)) if j>i]
        [axes[i][j].set_ylim(*axes[i][i].get_xlim()) for j in range(len(axes)) if j<i]
    
    # Labeling
    [ax_.set_xlabel(label_) for ax_, label_ in zip(axes[-1],labels)]
    [ax_.set_ylabel(label_) for ax_, label_ in zip(np.asarray(axes).T[0], labels)]
    [[ax_.set_yticklabels(["" for i in ax_.get_yticklabels()]) for ax_ in axc] for axc in np.asarray(axes).T[1:]]
    [[ax_.set_xticklabels(["" for i in ax_.get_xticklabels()]) for ax_ in axc] for axc in axes[:-1]]
    return axes
