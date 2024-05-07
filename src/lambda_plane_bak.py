#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:22:37 2024

@author: shelvon
@email: shelvonzang@gmail.com

"""

def plotRegions(axs, nklist=[1.5, 2.0, 1.0], kappa_modes=None):
    import sys
    import numpy as np
    import matplotlib.pyplot as _plt

    # for rotate arrow heads
    from matplotlib.markers import MarkerStyle
    from matplotlib.transforms import Affine2D

    # optical properties
    nbc = np.array([nklist[0], nklist[-1]]) # force to be a numpy array
    # nbc[0]: refractive index of the substrate
    # nbc[1]: refractive index of the cover

    n2bc = nbc**2
    
    # plotting parameters
    fill_modes = "full";
    alpha_modes = 1.0;
    ms = 10 # markersize
    alpha = 0.6
    
    cr = ["C0", "C1", "C2", "C3", "C5", "C7"] # color of each typical mode
    # cmaps = ["coolwarm", "BrBG", "PuOr", "Purples", "Reds", "Greys"] # color of each region
    # cmaps_r = ["coolwarm_r", "BrBG_r", "PuOr_r", "Purples_r", "Reds_r", "Greys_r"] # complementary color of each opposite region
    cmaps = ["Blues", "Oranges", "Purples", "Greys", "Reds", "Greens"] # color of each region
    # cmaps_r = ["Greens", "Reds", "Greys",  "Oranges", "Blues", "Purples"] # complementary color of each
    cmaps_r = ["Greys", "Reds", "Greens", "bone_r"] # complementary color of each
    
    # sampling in each kappa2 region
    nx = 71 # number of the sampling points along kappa2.real
    ny = 51 # number of the sampling points along kappa2.real
    offset = 1e-8 # a small shift from the branch cuts

    debugging = True
    # debugging = False
    if not debugging:
        nx *= 3
        ny *= 3

    # boundaries of each kappa2 region
    kappa2_re_max = max(n2bc)*5
    kappa2_re_min = -max(n2bc)*3
    kappa2_im_max = max(n2bc)*3

    n_re_max = max(np.asarray(nklist).real)*2.5
    n_re_min = -n_re_max
    n_im_max = n_re_max*3
    n_im_min = -n_im_max
    n2_re_max = kappa2_re_max
    n2_re_min = kappa2_re_min
    n2_im_max = kappa2_im_max
    n2_im_min = -n2_im_max

    zf_lambda = 1.0
    xmin = np.array([n_re_min, n_re_min*zf_lambda, n_re_min, n_re_min])
    xmax = np.array([n_re_max, n_re_max*zf_lambda, n_re_max, n_re_max])
    ymin = np.array([n_im_min, n_im_min*zf_lambda, n_re_min, n_re_min])
    ymax = np.array([n_im_max, n_im_max*zf_lambda, n_re_max, n_re_max])

    xlabels = ["$\\kappa '$", "$\\lambda '$", "$\\tau_s'$", "$\\tau_c'$"]
    ylabels = ["$\\kappa ''$", "$\\lambda ''$", "$\\tau_s''$", "$\\tau_c''$"]
    #%% create figure
    inch2cm = 1.0/2.54  # inches-->centimeters
    fontsize = 8
    figsize = 2*np.array([8.6, 6.4])*inch2cm

    _plt.rc('font', size=fontsize)

    # fig, _axs = _plt.subplots(figsize = figsize, ncols = 2, nrows = 2, constrained_layout = False);    
    # axs = _axs.flatten()
    
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)#, fillstyle="none")

    for i in range(4):
    
        # The following control lines are referred to 
        # https://stackoverflow.com/questions/13430231/how-i-can-get-cartesian-coordinate-system-in-matplotlib
    
        # labels
        axs[i].text(-0.05, 1.1, '('+chr(i+97)+')', transform=axs[i].transAxes, fontsize=fontsize*1.25, ha="left", fontweight="book")

        # Remove top and right spines
        # axs[i].spines['top'].set_visible(False)
        # axs[i].spines['right'].set_visible(False)

        # Set bottom and left spines as x and y axes of coordinate system
        # axs[i].spines['bottom'].set_position('zero')
        # if i==0:
        #     axs[i].spines['left'].set_position(('axes', 0))
        #     axs[i].plot((0), (1), marker='^', transform=axs[i].transAxes, **arrow_fmt)
        # else:
        #     axs[i].spines['left'].set_position('zero')
        #     axs[i].plot((0), (1), marker='^', transform=axs[i].get_xaxis_transform(), **arrow_fmt)

        # Draw arrows
        # axs[i].plot((1), (0), marker='>', transform=axs[i].get_yaxis_transform(), **arrow_fmt)
    
        # nticks = 3
        # if i==0:
        #     xtick_step = np.ceil((xmax[i]-xmin[i])/nticks*10)/10
        #     ytick_step = np.ceil((ymax[i]-ymin[i])/nticks*10)/10
        # else:        
        #     xtick_step = np.ceil((xmax[i]-xmin[i])/nticks)
        #     ytick_step = np.ceil((ymax[i]-ymin[i])/nticks)
    
        # x_ticks = np.arange(np.ceil(xmin[i]), np.floor(xmax[i])+1, xtick_step)
        # y_ticks = np.arange(np.ceil(ymin[i]), np.floor(ymax[i])+1, ytick_step)
        # axs[i].set_xticks(x_ticks[x_ticks != 0])
        # axs[i].set_yticks(y_ticks[y_ticks != 0])
            
        axs[i].set_xlabel(xlabels[i], usetex=True, size=fontsize*1.5, labelpad=-24, x=0.90, y=0.0, ha="left", va="center")
        axs[i].set_ylabel(ylabels[i], usetex=True, size=fontsize*1.5, labelpad=-28, x=0.0, y=0.82, rotation=0, ha="center")
    
        if i>=1:
            # axs[i].axline([0,0], [1,1], linestyle="--", linewidth=1, c="k")
            axs[i].axline([0,0], [-1,1], linestyle="--", linewidth=1, c="k")

        # axs[i].set_xticks([])
        # axs[i].set_yticks([])

        # axs[i].set_xlim([xmin[i], xmax[i]])
        # axs[i].set_ylim([ymin[i], ymax[i]])
    
        # if i==0:
        #     # axs[i].set_aspect(2.5, 'box')
        #     axs[i].set_aspect(2.0, 'datalim')
        # if i>0:
        #     # axs[i].set_aspect('equal', 'box')
        #     axs[i].set_aspect('equal', 'datalim')

    #%% branch cuts
    # kappa2_cut = np.array(
    #     [np.linspace(n2bc[0], n2bc[0] + 1j*kappa2_im_max, ny*3),
    #      np.linspace(n2bc[1], n2bc[1] + 1j*kappa2_im_max, ny*3)
    #      ])
    # kappa_cut = kappa2_cut**0.5

    kappa_cut = np.array(
        [np.linspace(nbc[0], nbc[0] + 1j*n_im_max, ny*3),
         np.linspace(nbc[1], nbc[1] + 1j*n_im_max, ny*3)
          ])
    kappa2_cut = kappa_cut**2

    def __plotCut(grid, bc):
        for icut in range(grid[:,0].size):
        
            # branch cuts in epsilon plane
            axs[0].plot(grid[icut,:].real, grid[icut,:].imag, "-", linewidth=1, c=cr[icut], ms=ms)
        
            # mark the media in the complex n and epsilon planes
            markerstyle = dict(marker="o", markeredgewidth=1, fillstyle="full", markerfacecolor='w', ms=5, zorder=1e4, clip_on=False)
            axs[0].plot(bc[icut].real, bc[icut].imag, c=cr[icut], **markerstyle)
        
            markerstyle = dict(marker="x", markeredgewidth=1, fillstyle="full", ms=3, zorder=1e6, clip_on=False, markerfacecolor='w')
            axs[0].plot(bc[icut].real, bc[icut].imag, c=cr[icut], **markerstyle)

    __plotCut(kappa2_cut, n2bc)
    #%% definition regions for kappa2 values
    
    def createGrid(x0, x1, y0, y1, nx=nx, ny=ny):
        x = np.linspace(x0+offset, x1-offset, nx)
        y = np.linspace(y0+offset, y1-offset, ny)

        x = np.pad(x, (1, 1), "constant", constant_values=(x0+0.5*offset, x1-0.5*offset))
        y = np.pad(y, (1, 1), "constant", constant_values=(y0+0.5*offset, y1-0.5*offset))

        # grid points of kappa2
        xx, yy = np.meshgrid(x, y, indexing="xy")
        cc = xx + 1j*yy

        # grid values of kappa2, for plotting the region with specified filling colors
        # zz = yy**0*0.01 # a uniform color
        zz = yy/np.max(np.abs(yy))*alpha+(1-alpha)

        # print(yy.shape, np.max(yy), np.max(zz))
        # different value (and color) for the boundaries
        # this is important for tricontourf
        zz[[0, -1], :] = 0
        zz[:, [0, -1]] = 0

        return cc, zz

    #### from kappa2 to kappa
    # kappa2grid_R1, zgrid_R1 = createGrid(n2bc[0], kappa2_re_max, 0, kappa2_im_max)
    # kappa2grid_R2, zgrid_R2 = createGrid(n2bc[1], n2bc[0], 0, kappa2_im_max)
    # kappa2grid_R3, zgrid_R3 = createGrid(kappa2_re_min, n2bc[1], 0, kappa2_im_max)
    # kappa2grid_R4, zgrid_R4 = createGrid(n2bc[0], kappa2_re_max, 0, -kappa2_im_max)
    # kappa2grid_R5, zgrid_R5 = createGrid(n2bc[1], n2bc[0], 0, -kappa2_im_max)
    # kappa2grid_R6, zgrid_R6 = createGrid(kappa2_re_min, n2bc[1], 0, -kappa2_im_max)

    # kappa2 values
    # kappa2grid = np.stack([kappa2grid_R1, kappa2grid_R2, kappa2grid_R3,
    #                        np.conj(kappa2grid_R1), np.conj(kappa2grid_R2), np.conj(kappa2grid_R3)])

    # zgrid = np.stack([zgrid_R1, zgrid_R2, zgrid_R3,
                      # -zgrid_R1, -zgrid_R2, -zgrid_R3])

    # # kappa values
    # kappagrid = kappa2grid**0.5
    
    #### from kappa to kappa2
    
    kappagrid_R1, zgrid_R1 = createGrid(nbc[0], n_re_max, 0, n_im_max)
    kappagrid_R2, zgrid_R2 = createGrid(nbc[1], nbc[0], 0, n_im_max)
    kappagrid_R3, zgrid_R3 = createGrid(0, nbc[1], 0, n_im_max)
    
    kappagrid = np.stack([kappagrid_R1, kappagrid_R2, kappagrid_R3])
    zgrid = np.stack([zgrid_R1, zgrid_R2, zgrid_R3])
    kappa2grid = kappagrid**2

    sign_kxleft = np.array([-1, 1, 1, 1, -1, -1])
    sign_kxright = np.array([-1, -1, 1, 1, 1, -1])

    # kx values
    kxleft = (n2bc[0]-kappa2grid)**0.5
    kxright = (n2bc[1]-kappa2grid)**0.5

    for ir in range(3):
        kxleft[ir] *= sign_kxleft[ir]
        kxright[ir] *= sign_kxright[ir]

    # lambda values
    lambdagrid = np.array([0.5*(kxleft + kxright),
                           0.5*(-kxleft + kxright),
                           0.5*(kxleft - kxright),
                           0.5*(-kxleft - kxright)])

    sign_cases = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
    
    # hatches = np.array(["++++", "////", "\\\\", "----"])
    #%% plot the definition regions
    # levels_cmap = np.array([0, 1]) # just two colors
    levels_cmap = np.linspace(0.1, 1.0, 21)
    
    levels_dense_cmap = np.linspace(0.1, 0.8, 21)

    # markerstyle = dict(marker="<", s=28, alpha=alpha_modes, zorder=1e8, clip_on=False)
    # axs[0].scatter(kappa_modes.real, kappa_modes.imag, c="gray", **markerstyle)

    # for ir in range(np.size(zgrid, 0)):
    for ir in range(3):

        # in the complex epsilon plane
        # axs[0].contourf(kappagrid[ir].real, kappagrid[ir].imag, zgrid[ir], cmap=cmaps[ir], levels=levels_dense_cmap, alpha=alpha)
        axs[0].contourf(kappa2grid[ir].real, kappa2grid[ir].imag, zgrid[ir], cmap=cmaps[ir], levels=levels_dense_cmap, alpha=alpha)

        # in the lambda plane
        for icase in range(4):
            axs[1].tricontourf(lambdagrid[icase,ir].real.flatten(), lambdagrid[icase,ir].imag.flatten(), zgrid[ir].flatten(), cmap=cmaps[ir], levels=levels_cmap)#, hatches=hatches[icase])
    
            # in the kxleft and kxright planes
            axs[2].tricontourf(sign_cases[icase, 0]*kxleft[ir].real.flatten(),
                                     sign_cases[icase, 0]*kxleft[ir].imag.flatten(),
                                     zgrid[ir].flatten(),
                                     cmap=cmaps[ir], levels=levels_cmap, alpha=alpha)
    
            axs[3].tricontourf(sign_cases[icase, 1]*kxright[ir].real.flatten(),
                                     sign_cases[icase, 1]*kxright[ir].imag.flatten(),
                                     zgrid[ir].flatten(),
                                     cmap=cmaps[ir], levels=levels_cmap, alpha=alpha)