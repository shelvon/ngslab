#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:22:37 2024

@author: shelvon
@email: xiaorun.zang@outlook.com

"""

# If nklist[0].real != nklist[-1].real and nklist[0].imag != nklist[-1].imag,
# then kappa values in the first quadrant for forward propagating and decaying modes are distributed in 9 regions, which are divided by two horizontal and two vertical straight lines passing through the two points: nklist[0] and nklist[-1] in the complex plane.

# def plotRegions(axs=None, Q=10, cut_by_kappa=True):

import sys
import numpy as np
import matplotlib.pyplot as plt

#%% define a few functions
def draw_Sx(ax, sign_Sx, sign_tau_im, xdir, x0, y0, w, h, color="k", lw=1.25):
    # sign_Sx: >0, arrow points to +x,
    #          <0, arrow points to -x.

    # xdir: +1, cover layer
    #       -1, substrate layer

    # sign_tau_im*xdir: >0, decaying, thus the density decreases away from the interface
    #                   <0, growing, thus the density increases away from the interface

    # centers of three arrows
    xy0 = np.array([x0, y0])
    x3 = np.array([0.25, 0.25, -0.25])*w*sign_tau_im*xdir + xy0[0]
    y3 = xy0[1] + np.array([-0.1, 0.1, 0])*h
    xy3 = np.array([x3, y3])

    # white background for the waveguide layer
    ax.fill([xy0[0]-0.5*w, xy0[0]+0.5*w, xy0[0]+0.5*w, xy0[0]-0.5*w],
            [xy0[1]-0.5*h, xy0[1]-0.5*h, xy0[1]+0.5*h, xy0[1]+0.5*h],
            color="w", edgecolor='w', alpha=1, clip_on=False)

    # arrow length
    al = 0.5*w

    # draw three arrows
    for ip in range(np.size(xy3, 1)):
        ax.annotate("",
                    xy=(xy3[0, ip]+al/2, xy3[1, ip]), xycoords='data',
                    xytext=(xy3[0, ip]-al/2, xy3[1, ip]), textcoords='data',
                    arrowprops=dict(arrowstyle=arrow_styles[sign_Sx], color=color,
                                    shrinkA=1, shrinkB=1,
                                    patchA=None, patchB=None,
                                    linewidth=lw*0.5,
                                    ),
                    )

    # draw the substrate or cover interface
    ax.plot([xy0[0]-xdir*0.5*w, xy0[0]-xdir*0.5*w],
            [xy0[1]-0.5*h, xy0[1]+0.5*h], "k-", linewidth=3)

def draw_Sy(ax, sign_Sy, sign_kappa_im, ydir, x0, y0, w, h, color="k", lw=1.25):
    # sign_Sy: >0, arrow points to +x,
    #          <0, arrow points to -x.

    # sign_kappa_im*ydir: >0, decaying, thus the density decreases during propagation
    #                     <0, growing, thus the density increases during propagation

    # centers of three arrows
    xy0 = np.array([x0, y0])
    x3 = np.array([-0.25, 0.25, 0])*w + xy0[0]
    y3 = xy0[1] + np.array([0.2, 0.2, -0.2])*h*sign_kappa_im*ydir
    xy3 = np.array([x3, y3])

    # white background for the waveguide layer
    ax.fill([xy0[0]-0.5*w, xy0[0]+0.5*w, xy0[0]+0.5*w, xy0[0]-0.5*w],
            [xy0[1]-0.5*h, xy0[1]-0.5*h, xy0[1]+0.5*h, xy0[1]+0.5*h],
            color="w", edgecolor='w', alpha=1, clip_on=False)

    # draw the substrate or cover interface
    ax.plot([xy0[0]-0.5*w, xy0[0]-0.5*w],
            [xy0[1]-0.5*h, xy0[1]+0.5*h], "C7-", linewidth=3)
    ax.plot([xy0[0]+0.5*w, xy0[0]+0.5*w],
            [xy0[1]-0.5*h, xy0[1]+0.5*h], "C7-", linewidth=3)

    # arrow length
    al = 0.5*h

    # print(sign_kappa_im*ydir)
    # draw three arrows
    for ip in range(np.size(xy3, 1)):
        ax.annotate("",
                    xy=(xy3[0, ip], xy3[1, ip]+al/2), xycoords='data',
                    xytext=(xy3[0, ip], xy3[1, ip]-al/2), textcoords='data',
                    arrowprops=dict(arrowstyle=arrow_styles[sign_Sy], color=color,
                                    shrinkA=1, shrinkB=1,
                                    patchA=None, patchB=None,
                                    linewidth=lw,
                                    ),
                    )

def draw_arc(ax, x, y, color="k"):
    ax.annotate("",
                xy=(x[0], y[0]), xycoords='data',
                xytext=(x[1], y[1]), textcoords='data',
                arrowprops=dict(arrowstyle="->", color=color,
                                shrinkA=1, shrinkB=1,
                                patchA=None, patchB=None,
                                linewidth=1.5,
                                connectionstyle="arc3,rad=0.7",
                                ),
                )

#%% plotting parameters
inch2cm = 1.0/2.54  # inches-->centimeters
zf = 1.0
figsize = [zf*2*8.6*inch2cm, zf*8*inch2cm]
plt.close("all")

fig, _axs = plt.subplots(figsize = figsize, ncols = 4, nrows = 2, constrained_layout = True);
axs = _axs.flatten()

# sampling in each kappa2 region
nx = 11 # number of the sampling points along kappa2.real
ny = 11 # number of the sampling points along kappa2.real
offset = 1e-6 # a small shift from the branch cuts, or branch points

Q = 10
cut_by_kappa = True
# the branch cuts are vertical lines going through
cut_by_kappa2 = not cut_by_kappa

debugging = True
# debugging = False
if not debugging:
    # more refined sampling points
    nx *= 3
    ny *= 3

n_re_max = 5
n_re_min = 0
n_im_max = 1
n_im_min = 0
n2_re_min = 0
n2_re_max = n_re_max**2
n2_im_min = 0
n2_im_max = n2_re_max

xmin = np.array([-n_re_max, -n_re_max, -n_re_max, -n_re_max,
                 -n2_re_max, -n_re_max, -n_re_max, -n_re_max])
xmax = np.array([n_re_max, n_re_max, n_re_max, n_re_max,
                 n2_re_max, n_re_max, n_re_max, n_re_max])

ymin = np.array([-n_im_max, -n_im_max, -n_im_max, -n_im_max,
                 -n2_im_max, -n_im_max, -n_im_max, -n_im_max])
ymax = np.array([n_im_max, n_im_max, n_im_max, n_im_max,
                 n2_im_max, n_im_max, n_im_max, n_im_max])

n_re_step = (n_re_max-n_re_min)/nx
n_im_step = (n_im_max-n_im_min)/ny

xlabels = ["$n'$", "$\\kappa '$", "$\\tau_\mathrm{s}'$", "$\\tau_\mathrm{c}'$",
           "$\\epsilon '$", "$\\kappa '$", "$\\tau_\mathrm{s}'$", "$\\tau_\mathrm{c}'$"]
ylabels = ["$n''$", "$\\kappa ''$", "$\\tau_\mathrm{s}''$", "$\\tau_\mathrm{c}''$",
           "$\\epsilon ''$", "$\\kappa ''$", "$\\tau_\mathrm{s}''$", "$\\tau_\mathrm{c}''$"]

#%% create subplots
fontsize = 8
ms = 10 # markersize
nticks = 1
arrow_fmt = dict(markersize=3, color='black', clip_on=False)#, fillstyle="none")

for i in range(8):
    # labels
    axs[i].text(-0.05, 1.1, '('+chr(i+97)+')', transform=axs[i].transAxes, fontsize=fontsize*1.25, ha="left", fontweight="book")

    if i==4:
        axs[i].set_xlabel(xlabels[i], usetex=True, size=fontsize*1.5, labelpad=-6, x=1.02, ha="left", va="center")
    else:
        axs[i].set_xlabel(xlabels[i], usetex=True, size=fontsize*1.5, labelpad=-8, x=0.92, ha="left", va="center")
    axs[i].set_ylabel(ylabels[i], usetex=True, size=fontsize*1.5, labelpad=-5, y=1.08, rotation=0, ha="center", va="center")

    axs[i].spines['bottom'].set_position('zero')
    axs[i].spines['left'].set_position('zero')
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    # axs[i].spines['bottom'].set(zorder=-1)
    # axs[i].spines['left'].set(zorder=-1)

    # xtick_step = np.ceil((xmax[i]-xmin[i])/nticks)
    # ytick_step = np.ceil((ymax[i]-ymin[i])/nticks)
    # x_ticks = np.arange(np.ceil(xmin[i]), np.floor(xmax[i])+1, xtick_step)
    # y_ticks = np.arange(np.ceil(ymin[i]), np.floor(ymax[i])+1, ytick_step)
    # axs[i].set_xticks(x_ticks[x_ticks != 0])
    # axs[i].set_yticks(y_ticks[y_ticks != 0])

    axs[i].set_xticks([])
    axs[i].set_yticks([])

    # Draw arrows for coordinate axes
    axs[i].plot((0.98), (0), marker='>', transform=axs[i].get_yaxis_transform(), **arrow_fmt)
    axs[i].plot((0), (0.98), marker='^', transform=axs[i].get_xaxis_transform(), **arrow_fmt)

    axs[i].set_aspect("equal", "datalim")

#%% styles of different modes
n_samples = np.array([(n_re_max+1j*n_re_max/Q/1.5)*0.6, (n_re_max/Q+1j*n_re_max)*0.8])
epsilon_samples = n_samples**2
kappa_samples = -(1 - 1j*epsilon_samples.real/epsilon_samples.imag)

Q_samples = np.abs(epsilon_samples.real/epsilon_samples.imag)
zf_samples = Q_samples/Q_samples[1]

taus_samples = kappa_samples
tauc_samples = kappa_samples

# branch cuts in n plane
n_samplesPlot = np.array([np.linspace(0, n_samples[0], nx),
                          np.linspace(0, n_samples[1], nx)])

epsilon_samplesPlot = n_samplesPlot**2

# hatches = ["///", "\\\\\\", "xxx", "++"]
# hatches = ["xxxx", ".", ".", "."]
# colors = ["C1", "C2", "C4", "C5"]
hatch_del = "xxx"
# hatches = ["///", "\\\\\\", "///", "\\\\\\"]
hatches = ["", "", "", ""]
# hatches = ["///", "xxx", "///", "xxx"]
# hatches = ["...", "...", "...", "..."]
# hatches = ["xxxx", "xxxx", "xxxx", "xxxx"]
# hatches = ["+++", "+++", "+++", "+++"]
color_del = "C7"
colors = ["C3", "C3", "C2", "C2"]
colors_tau = ["C3", "C3", "C2", "C2"]
# colors_tau = ["C4", "C4", "C7", "C7"]
arrow_styles = ["", "->", "<-"] #Sx>0, or Sx<0

#%% plot different mode types
medium_labels = ["dielectric", "metal"]
kappa_regions =[["", "", ""], ["", "U", "V"], ["", "W", "X"]]
tau_regions = [["", "", ""], ["", "B", "A"], ["", "C", "D"]]

for im in range(np.size(n_samplesPlot,0)):
    # n plane
    axs[0].plot(n_samplesPlot[im, :].real, n_samplesPlot[im, :].imag, "--", linewidth=1, c="k", ms=3)
    axs[0].plot(n_samples[im].real, n_samples[im].imag, "o", linewidth=1, c="C"+str(im), ms=4)
    axs[0].text(n_samples[im].real+0.1*n_re_max, n_samples[im].imag+0.1*n_re_max,
                medium_labels[im],
                color="C"+str(im), ha="center", va="center", fontsize=fontsize)

    # epsilon plane
    axs[4].plot(epsilon_samplesPlot[im, :].real, epsilon_samplesPlot[im, :].imag, "--", linewidth=1, c="k", ms=3)
    axs[4].plot(epsilon_samples[im].real, epsilon_samples[im].imag, "o", linewidth=1, c="C"+str(im), ms=4)
    axs[4].text(epsilon_samples[im].real*0.8, epsilon_samples[im].imag+0.12*n2_re_max,
                medium_labels[im],
                color="C"+str(im), ha="center", va="center", fontsize=fontsize)

    # kappa plane
    sign = np.sign(epsilon_samples[im].real)*zf_samples[im]
    kappa_im_abs = np.abs(kappa_samples[im].imag)
    kappa_re_abs = np.abs(kappa_samples[im].real)
    # Sy>0, kappa.imag>0
    axs[im*4+1].fill([0, sign*n_re_max, sign*n_re_max, -1*sign*kappa_re_abs],
                     [0, 0, kappa_im_abs, kappa_im_abs],
                    hatch=hatches[0],
                    color=colors[0], alpha=0.5, edgecolor='w', clip_on=False)
    # Sy>0, kappa.imag<0
    axs[im*4+1].fill([sign*kappa_re_abs, sign*n_re_max, sign*n_re_max, 0],
                     [-kappa_im_abs, -kappa_im_abs, 0, 0],
                     hatch=hatches[1],
                     color=colors[1], alpha=0.5, edgecolor='w', clip_on=False)
    # Sy<0, kappa.imag>0
    axs[im*4+1].fill([0, -1*sign*n_re_max, -1*sign*n_re_max, -1*sign*kappa_re_abs],
                     [0, 0, kappa_im_abs, kappa_im_abs],
                    hatch=hatches[2],
                    color=colors[2], alpha=0.5, edgecolor='w', clip_on=False)
    # Sy<0, kappa.imag<0
    axs[im*4+1].fill([sign*kappa_re_abs, -1*sign*n_re_max, -1*sign*n_re_max, 0],
                     [-kappa_im_abs, -kappa_im_abs, 0, 0],
                    hatch=hatches[3],
                    color=colors[3], alpha=0.5, edgecolor='w', clip_on=False)

    axs[im*4+1].plot([-kappa_samples[im].real*zf_samples[im], kappa_samples[im].real*zf_samples[im]],
                     [-kappa_samples[im].imag, kappa_samples[im].imag],
                     "-", linewidth=1.5, c="C"+str(im), ms=3)
    # axs[im*4+1].arrow(0, 0, kappa_samples[im].real*zf_samples[im], kappa_samples[im].imag,
    #                   shape='full', lw=0, length_includes_head=True, head_width=.6*zf_samples[im], facecolor="C"+str(im), alpha=1, clip_on=False)
    # Sy=0
    # axs[im*4+1].text(-n_re_max*zf_samples[im]*0.75, kappa_samples[im].imag*1.2,
    #                  "$\\bar{S}_y^{\\mathrm{(TM)}}=0$",
    #                  color="C"+str(im), ha="left", va="center", fontsize=fontsize)

    axs[im*4+1].text(np.sign(kappa_samples[im].imag)*1.5, -np.abs(kappa_samples[im].imag)*1.28,
                      "$\\bar{S}_y^{\\mathrm{(TM)}}=0$",
                      color="C"+str(im), ha="center", va="center", fontsize=fontsize)

    # taus plane
    # sign = np.sign(epsilon_samples[im].real)*zf_samples[im]
    taus_im_abs = np.abs(kappa_samples[im].imag)
    taus_re_abs = np.abs(kappa_samples[im].real)
    # Sx>0, incoming to the wg, taus.imag<0 decaying
    axs[im*4+2].fill([sign*taus_re_abs, sign*n_re_max, sign*n_re_max, 0],
                     [-taus_im_abs, -taus_im_abs, 0, 0],
                    hatch=hatches[0],
                    color=colors_tau[0], alpha=0.5, edgecolor='w', clip_on=False)
    # Sx>0, incoming to the wg, taus.imag>0 growing
    axs[im*4+2].fill([0, sign*n_re_max, sign*n_re_max, -1*sign*taus_re_abs],
                     [0, 0, taus_im_abs, taus_im_abs],
                    # hatch=hatch_del,
                    color=color_del, alpha=0.5, edgecolor='w', clip_on=False)
    # Sx<0, outgoing from the wg, taus.imag<0 decaying
    axs[im*4+2].fill([sign*taus_re_abs, -1*sign*n_re_max, -1*sign*n_re_max, 0],
                     [-taus_im_abs, -taus_im_abs, 0, 0],
                    hatch=hatches[2],
                    color=colors_tau[2], alpha=0.5, edgecolor='w', clip_on=False)
    # Sx<0, outgoing from the wg, taus.imag>0 growing
    axs[im*4+2].fill([0, -1*sign*n_re_max, -1*sign*n_re_max, -1*sign*taus_re_abs],
                     [0, 0, taus_im_abs, taus_im_abs],
                    hatch=hatches[3],
                    color=colors_tau[3], alpha=0.5, edgecolor='w', clip_on=False)

    axs[im*4+2].plot([-taus_samples[im].real*zf_samples[im], taus_samples[im].real*zf_samples[im]],
                     [-taus_samples[im].imag, taus_samples[im].imag],
                     "-", linewidth=1.5, c="C"+str(im), ms=3)
    # axs[im*4+2].arrow(0, 0, taus_samples[im].real*zf_samples[im], taus_samples[im].imag,
                      # shape='full', lw=0, length_includes_head=True, head_width=.6*zf_samples[im], facecolor="C"+str(im), alpha=1, clip_on=False)
    # Sx=0
    axs[im*4+2].text(np.sign(kappa_samples[im].imag)*1.5, -np.abs(kappa_samples[im].imag)*1.28,
                     "$\\bar{S}_x^{\\mathrm{(TM)}}=0$",
                     color="C"+str(im), ha="center", va="center", fontsize=fontsize)

    # tauc plane
    # sign = np.sign(epsilon_samples[im].real)*zf_samples[im]
    tauc_im_abs = np.abs(kappa_samples[im].imag)
    tauc_re_abs = np.abs(kappa_samples[im].real)
    # Sx<0, incoming to the wg, taus.imag>0 decaying
    axs[im*4+3].fill([0, -1*sign*n_re_max, -1*sign*n_re_max, -1*sign*taus_re_abs],
                     [0, 0, taus_im_abs, taus_im_abs],
                    hatch=hatches[2],
                    color=colors_tau[2], alpha=0.5, edgecolor='w', clip_on=False)
    # Sx<0, incoming to the wg, taus.imag<0 growing
    axs[im*4+3].fill([sign*taus_re_abs, -1*sign*n_re_max, -1*sign*n_re_max, 0],
                     [-taus_im_abs, -taus_im_abs, 0, 0],
                    # hatch=hatch_del,
                    color=color_del, alpha=0.5, edgecolor='w', clip_on=False)
    # Sx>0, outgoing from the wg, taus.imag>0 decaying
    axs[im*4+3].fill([0, sign*n_re_max, sign*n_re_max, -1*sign*taus_re_abs],
                     [0, 0, taus_im_abs, taus_im_abs],
                    hatch=hatches[0],
                    color=colors_tau[0], alpha=0.5, edgecolor='w', clip_on=False)
    # Sx>0, outgoing from the wg, taus.imag<0 growing
    axs[im*4+3].fill([sign*tauc_re_abs, sign*n_re_max, sign*n_re_max, 0],
                     [-tauc_im_abs, -tauc_im_abs, 0, 0],
                    hatch=hatches[1],
                    color=colors_tau[1], alpha=0.5, edgecolor='w', clip_on=False)

    axs[im*4+3].plot([-tauc_samples[im].real*zf_samples[im], tauc_samples[im].real*zf_samples[im]],
                     [-tauc_samples[im].imag, tauc_samples[im].imag],
                     "-", linewidth=1.5, c="C"+str(im), ms=3)
    # axs[im*4+3].arrow(0, 0, tauc_samples[im].real*zf_samples[im], tauc_samples[im].imag,
    #                   shape='full', lw=0, length_includes_head=True, head_width=.6*zf_samples[im], facecolor="C"+str(im), alpha=1, clip_on=False)
    # Sx=0
    axs[im*4+3].text(np.sign(kappa_samples[im].imag)*1.5, -np.abs(kappa_samples[im].imag)*1.28,
                     "$\\bar{S}_x^{\\mathrm{(TM)}}=0$",
                     color="C"+str(im), ha="center", va="center", fontsize=fontsize)

    # ic==1, kappa: Sy>0, tau: Sx>0
    # ic==-1, kappa: Sy<0, tau: Sx<0
    # for ic in [1, -1]:
    #     for ia in range(1, 4):
    #         draw_arc(axs[im*4+ia],
    #                   [-ic*kappa_samples[im].real*zf_samples[im], ic*kappa_samples[im].real*zf_samples[im]],
    #                   [-ic*kappa_samples[im].imag, ic*kappa_samples[im].imag],
    #                   color=colors[(ic+1)])

    Q_threshold = 10
    axs[im*4+1].plot([-n_re_max*zf_samples[im], n_re_max*zf_samples[im]],
                     [-n_re_max*zf_samples[im]/Q_threshold, n_re_max*zf_samples[im]/Q_threshold],
                     'C7--')
    axs[im*4+1].plot([n_re_max*zf_samples[im], -n_re_max*zf_samples[im]],
                     [-n_re_max*zf_samples[im]/Q_threshold, n_re_max*zf_samples[im]/Q_threshold],
                     'C7--')

    axs[im*4+1].text(n_re_max*zf_samples[im], -n_re_max*zf_samples[im]/Q_threshold,
                     "$\\left|\\frac{\\kappa'}{\\kappa''}\\right|=$"+str(Q_threshold),
                     # fontweight="bold",
                     color="C7", ha="left", va="top", fontsize=fontsize)

    for iSy in [+1, -1]:
        for ikappa_im in [+1, -1]:
            sign_Sy = int(iSy*np.sign(epsilon_samples[im].real))
            draw_Sy(axs[im*4+1], sign_Sy=sign_Sy, sign_kappa_im=ikappa_im, ydir=+1,
                    x0=n_re_max*0.6*zf_samples[im]*iSy,
                    y0=-n_re_max*0.55*zf_samples[im]*ikappa_im,
                    w=n_re_max*0.4*zf_samples[im],
                    h=n_re_max*0.7*zf_samples[im],
                    color=colors[sign_Sy])
            axs[im*4+1].text(n_re_max*0.2*zf_samples[im]*iSy, n_re_max*0.2*zf_samples[im]*ikappa_im,
                             # "$"+kappa_regions[ikappa_im*(1-im*2)][sign_Sy]+"$",
                             "$\\mathrm{"+kappa_regions[ikappa_im][sign_Sy]+"}$",
                             color="k", ha="center", va="center", fontsize=fontsize*1.25,
                             )

    for iSx in [+1, -1]:
        for itau_im in [+1, -1]:
            sign_Sx = int(iSx*np.sign(epsilon_samples[im].real))
            draw_Sx(axs[im*4+2], sign_Sx=sign_Sx, sign_tau_im=itau_im, xdir=-1,
                    x0=n_re_max*0.6*zf_samples[im]*iSx,
                    y0=n_re_max*0.6*zf_samples[im]*itau_im,
                    w=n_re_max*0.4*zf_samples[im],
                    h=n_re_max*0.6*zf_samples[im],
                    color=colors_tau[sign_Sx])
            axs[im*4+2].text(n_re_max*0.2*zf_samples[im]*iSx, n_re_max*0.2*zf_samples[im]*itau_im,
                             "$\\mathrm{"+tau_regions[-itau_im][-sign_Sx]+"}_\mathrm{s}$",
                             color="k", ha="center", va="center", fontsize=fontsize*1.25,
                             )

            draw_Sx(axs[im*4+3], sign_Sx=sign_Sx, sign_tau_im=itau_im, xdir=+1,
                    x0=n_re_max*0.6*zf_samples[im]*iSx,
                    y0=-n_re_max*0.6*zf_samples[im]*itau_im,
                    w=n_re_max*0.4*zf_samples[im],
                    h=n_re_max*0.6*zf_samples[im],
                    color=colors_tau[sign_Sx])
            axs[im*4+3].text(n_re_max*0.2*zf_samples[im]*iSx, n_re_max*0.2*zf_samples[im]*itau_im,
                             "$\\mathrm{"+tau_regions[itau_im][sign_Sx]+"}_\mathrm{c}$",
                             color="k", ha="center", va="center", fontsize=fontsize*1.25,
                             )

# two regions for n and epsilon planes
axs[0].fill([0, n_re_max, n_re_max, 0], [0, 0, n_re_max, 0],
            color="C0", alpha=0.2, edgecolor='w', clip_on=False)
axs[0].fill([0, n_re_max, 0, 0], [0, n_re_max, n_re_max, 0],
            color="C1", alpha=0.2, edgecolor='w', clip_on=False)

axs[4].fill([0, n2_re_max*0.4, n2_re_max*0.4, 0], [0, 0, n2_re_max*0.8, n2_re_max*0.8],
            color="C0", alpha=0.2, edgecolor='w', clip_on=False)
axs[4].fill([0, -n2_re_max*0.8, -n2_re_max*0.8, 0], [0, 0, n2_re_max*0.8, n2_re_max*0.8],
            color="C1", alpha=0.2, edgecolor='w', clip_on=False)

sys.exit(0)
#%% save the figure
figname = "4_complex_plane";
# plt.savefig(figname + '.png',format='png', dpi=300)
plt.savefig(figname + '.pdf',format='pdf')
