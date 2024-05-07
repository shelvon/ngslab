#!/usr/bin/env python3
"""
@author: shelvon
@email: xiaorun.zang@outlook.com

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

import copy

# For a local debugging, we import or install the local package.
#
# - Method 1, append the folder's path of the local package:

try:
    import ngslab as slab
except:
    srcpath = os.path.dirname(
        os.path.dirname(
            os.path.abspath(os.getcwd())
            )
        )
    sys.path.append(srcpath)
    import ngslab as slab
#
# - Method 2, pip install locally the python package, where "." is intended
# to be run under the root folder of the python package.
# python3 -m pip install --upgrade.
#
# - Method 3, conda install the local package build by conda-build.
# conda install path/to/local/package

# import slab

# print(dir()) # print what have been imported

# sys.exit(0)
#%% silver-silica slab waveguide:
def wg_AgSiO2(geom_tSlab):
    nSlab = np.maximum(geom_nPML, round(np.ceil(geom_tSlab/geom_hMax)))
    wg = dict(
        # geometry
        name = "demo_AgSiO2",
        intervals = [-(geom_tPML + geom_tSub + geom_tSlab), -(geom_tSub + geom_tSlab), -geom_tSlab, 0.0, geom_tFree, geom_tFree + geom_tPML],
        labels = ["pml_left", "sub", "slab", "cover", "pml_right"],
        nnodes = [geom_nPML, round(np.ceil(geom_tSub/geom_hMax)), round(np.ceil(geom_tSlab/geom_hMax)), round(np.ceil(geom_tFree/geom_hMax)), geom_nPML],
        # material
        nk = {"default":1.0, "air": 1.0, "SiO2": 1.4537, "Si3N4": 2.0255, "silver": "agjc", "silverDrude":"agDrude"},
        map = {"pml_left":"silver", "sub":"silver", "slab":"SiO2"},
        # map = {"pml_left":"silverDrude", "sub":"silverDrude", "slab":"SiO2"},
        )

    return wg

#%% model creation
#---- Geometric Parameters
model = slab.SlabWaveGuide();
# the wavelength of the angular plasma frequency
# for silver is
# 2*np.pi*299792458/omega_p, where omega_p = 1.3688599705386422e+16 (rad/s)
model.ld0_target = 1.376073234552722e-07

# use micrometer may yield lower condition number and better accuracy
model.ld0_target *= 1e6

# The thicknesses geom_tFree and geom_tSub become irrelevant
# to numerical accuracy, if TBCs are used.
geom_tFree = model.ld0_target*0.3
geom_tSub = model.ld0_target*0.3
geom_tPML = model.ld0_target*0.2

# geom_hMax = model.ld0_target/31; geom_nPML = 13; # extra fine mesh
# geom_hMax = model.ld0_target/31; geom_nPML = 27; # extra fine PML mesh
geom_hMax = model.ld0_target/17; geom_nPML = 1; # TBC debugging

geom_delPML = geom_tPML/geom_nPML
print("hMax="+f"{geom_hMax/model.ld0_target:.4f}*ld0")
print("hPML="+f"{geom_delPML/model.ld0_target:.4f}*ld0")

#%% loop in the slab thickness
tSlabArray = np.arange(200e-9, 201e-9, 20e-9)

# Here, a change of unit to micrometer is needed as well.
tSlabArray *= 1e6
for it, geom_tSlab in zip(range(tSlabArray.size), tSlabArray):
    print("tSlab["+str(it)+"] = "+f"{geom_tSlab*1e3:.0f} nm")

    # choose a waveguide
    wg = wg_AgSiO2(geom_tSlab)

    # update the waveguide geometry
    model.geom.Update(wg["intervals"], wg["nnodes"], wg["labels"])

    #-- create the mesh
    model.mesh.Create()
    # model.mesh.Plot();

    model.material.nk = wg["nk"]
    model.material.map = wg["map"]

    # enable PML domains
    # model.material.pml = {"pml_left":True, "pml_right":True}
    # disable PML
    # model.material.pml = {"pml_left":False, "pml_right":False} # False by default

    # and then set the transparent boundary condition
    # setTBC() accepts two numbers defining the indices of the leftmost to the rightmost semi-infinite regions that bounding the core waveguide region
    model.setTBC(tbc_doms=[1, -2]) # by default: tbc_doms=[1, -2]
    # model.setTBC(tbc_doms=[0, -1]) # test

    # model.mesh.Plot()
    # sys.exit(0)
    #%% model build
    #---- build the model (create function space, set up materials, enable pml materials)
    # bspl_order: the order of the ngsolve.BSpline representations for dispersive and tabulated materials
    model.Build(fes_order=2, bspl_order=2, ld_scale=1e6) # bspl_order = 1 (linear), 2 (quadratic) curve

    # debug PML
    # model.SetPML(model.material.pml, pml_plot=True);

    # sys.exit(0)

    #%% figure properties
    plt.close("all")
    inch2cm = 1.0/2.54  # inches-->centimeters
    figsize = 2.0*np.array([8.6, 5.4])*inch2cm
    ms = 4
    fontsize = 8
    plt.rc('font', size=fontsize)

    ll_plotted = False # plot light lines
    c_fem = ["g", "C7"]
    alpha_fem = [1.0, 0.6]
    markerstyle_fem = dict(marker="o", s=round(ms*2), edgecolor="none", clip_on=True)# ,zorder=1)

    fig = plt.figure(figsize = figsize, layout="constrained")
    gs = GridSpec(4, 6, figure=fig)

    axSpectra = fig.add_subplot(gs[0:2, 0:2])
    ax_hz1 = fig.add_subplot(gs[0, 2:4])
    ax_hz2 = fig.add_subplot(gs[0, 4:6])
    ax_hz3 = fig.add_subplot(gs[1, 2:4])
    ax_hz4 = fig.add_subplot(gs[1, 4:6])
    ax_kappa = fig.add_subplot(gs[2:4, 0:2])
    ax_taus = fig.add_subplot(gs[2:4, 2:4])
    ax_tauc = fig.add_subplot(gs[2:4, 4:6])

    axs = [axSpectra, ax_hz1, ax_hz2, ax_hz3, ax_hz4, ax_kappa, ax_taus, ax_tauc]
    for i in range(len(axs)): # create (a), (b), ... labels for each subplot
        x = -0.03
        y = 1.08
        if i==0:
            y = 1.03
        axs[i].text(x, y, '('+chr(i+97)+')', transform=axs[i].transAxes,
                    fontsize=fontsize*1.25, ha="right", va="bottom", fontweight="book", fontname="Sans")

    # sys.exit(0)
    #%% model simulation

    #-- normalized frequencies
    model.wArray = np.array([0.37])
    model.wArray = np.concatenate( (model.wArray, np.arange(0.1, 0.5, 0.01)) )
    model.wArray = np.concatenate( (model.wArray, np.arange(0.3, 0.5, 0.005)) )
    model.wArray = np.unique(np.round(model.wArray*1000))/1000 # keep unique floating numbers of 3 decimal precision.

    # print("ldArray="+str(model.ld0_target/model.wArray)+" um")
    for iw in range(model.wArray.size):
    # for iw in range(model.wArray.size)[::-1]: # in reverse order
        model.w = model.wArray[iw]

        sol = model.Solve(show_pattern=False)

        # sys.exit(0)
        #%% investigate the eigenvalues

        n2list = np.asarray(model.sol._n2list)
        nlist = np.asarray(n2list)**0.5
        ns, nc = nlist[[0, -1]]
        n2s, n2c = n2list[[0, -1]]

        #### filtering modes
        mode_markers = ["o", "^"]
        Q_threshold = 1
        # Q_threshold = (1/np.e)
        # Q_threshold = 1e-1
        kappa_tol = 1e-12
        kappa2_limit = np.max(np.abs(n2list))*100.1

        # the last one will be plotted in the complex planes
        # mode_type == 0: TM mode; mode_type == 1: TE mode
        for mode_type in [0]:
        # for mode_type in [1, 0]:
        # for mode_type in [0, 1]:

            ## results from FEM simulations
            nvalid = model.sol._nvalid[mode_type]
            kappa_fem = model.sol.kappa[mode_type, :nvalid]
            kappa2_fem = model.sol.kappa2[mode_type, :nvalid]
            taus_fem = model.sol.taus[mode_type, :nvalid]
            tauc_fem = model.sol.tauc[mode_type, :nvalid]

            # filter on kappa
            idx_kappa_fem = np.where(
                ~np.isnan(kappa_fem)
                & (np.abs(kappa_fem.real/kappa_fem.imag)>Q_threshold)
                & (np.abs(kappa_fem)**2<kappa2_limit)
                )[0]

            # divided tau plane into regions according to the sign of Sx
            if mode_type==0: # TM mode
                Sx_s = (taus_fem.real*n2s.real+taus_fem.imag*n2s.imag)
                Sx_c = (tauc_fem.real*n2c.real+tauc_fem.imag*n2c.imag)
            elif mode_type==1: # TE mode for nonmagnetic material with (mu.real = 1, mu.imag=0)
                Sx_s = -(taus_fem.real)
                Sx_c = -(tauc_fem.real)

            # Keep modes that do not have energy feeding into the waveguide region from the infinity
            # That means, the situations of both the power flow direction points into the waveguide
            # and density increases away from the waveguide cannot be met simultaneously.
            #FIXME: According to the following paper
                # [1] T. Tamir and F. Kou, ‘Varieties of leaky waves and their excitation along multilayered structures’, IEEE Journal of Quantum Electronics, vol. 22, no. 4, pp. 544–551, Apr. 1986, doi: 10.1109/JQE.1986.1072991.
                # The contra-leaky mode V cannot be excited by any realistic sources.
                # However, this argument is not so obvious without giving any further explanation in that paper.
            idx_s = np.where(
                ( ~((Sx_s>-kappa_tol) & (taus_fem.imag>-kappa_tol)) )
                )[0]
            idx_c = np.where(
                ( ~((Sx_c<kappa_tol) & (tauc_fem.imag<kappa_tol)) )
                )[0]

            # mode indices of power flux from the denser cladding layer to the rarer cladding layer
            if ns.real>=nc.real:
                idx_denser2rarer = np.where(
                    (~(
                        ((Sx_s>-kappa_tol) & (taus_fem.imag<kappa_tol)) # power flow type As
                        &((Sx_c>-kappa_tol) & (tauc_fem.imag<kappa_tol)) # power flow type Cc
                       ))
                    )[0]

            idx = np.intersect1d(idx_s, idx_c)
            # idx = np.intersect1d(idx, idx_denser2rarer)
            idx = np.intersect1d(idx_kappa_fem, idx)
            # idx = idx_denser2rarer

            # idx = idx_kappa_fem
            idx_del = np.setdiff1d(np.arange(nvalid), idx)

            if idx.size>0:

                # modes selected by all three conditions
                axSpectra.scatter(
                    np.abs(kappa_fem[idx].real)*model.k0, np.repeat(model.w, idx.size),
                    # c = "C0",
                    c=np.full(idx.size, 1),
                    # c=range(idx.size),
                    # cmap="viridis_r",
                    cmap="seismic",
                    s=ms,
                    alpha=np.exp(-np.abs(kappa2_fem[idx].imag)-kappa_tol)**0
                    )

                # dispersive light line(s) is plotted wavelength by wavelength
                # axSpectra.scatter(np.abs(nlist[0].real)*model.k0, model.w, s=1, c = "gray")

        # plot two light lines once
        if not ll_plotted:

            axSpectra.axline((0,0), (nlist[1]*2*np.pi/model.ld0Array[0], model.wArray[0]), c="gray")
            axSpectra.axline((0,0), (nlist[2]*2*np.pi/model.ld0Array[0], model.wArray[0]), c="gray")

            ll_plotted = True

        # continue
        #### at a single frequency
        # investigate eigenvalues in the complex planes
        if (model.w==0.37):
            sol_probe = copy.deepcopy(sol)

            cmap_fem = "viridis_r"
            cmap_fem_del = "Grays"
            c_hz = ["C0", "C1", "C3", "C9"]

            axsScatter = []
            for ax, zz in zip([ax_kappa, ax_taus, ax_tauc],
                              [sol_probe.kappa[mode_type, :sol_probe._nvalid[mode_type]],
                               sol_probe.taus[mode_type, :sol_probe._nvalid[mode_type]],
                               sol_probe.tauc[mode_type, :sol_probe._nvalid[mode_type]],
                               ]):

                ax.scatter(zz[idx_del].real, zz[idx_del].imag, c=c_fem[1], alpha=alpha_fem[1], **markerstyle_fem)
                # markerstyle_fem["zorder"] += 1
                axsScatter.append(ax.scatter(zz[idx].real, zz[idx].imag, c=range(idx.size)[::-1], alpha=alpha_fem[0], **markerstyle_fem))

            # extract the field profile of selected modes
            axs_names = ["kappa", "taus", "tauc"]
            axs_select = np.asarray(axsScatter)

            # plt.close(1) # close the figure for plotting the dispersion curve
            ####  interactively select a mode by clicking at the nearest point in the complex plane
            mode_selector = slab.SlabWaveGuide.modeSelector(
                axs_select, sol_probe, axs_names, mode_type=mode_type, nmodes = 4,
                axSpectra=axSpectra, axs_profile=np.array([ax_hz1, ax_hz2, ax_hz3, ax_hz4]))

            # plot selected modes
            # mode_selector.mpl_connect()
            mode_selector.plotModes(idx[3])
            mode_selector.plotModes(idx[2])
            mode_selector.plotModes(idx[1])
            mode_selector.plotModes(idx[0])

            kappa2_max = np.max(np.abs(np.asarray(sol_probe._n2list)))
            kappa_max = kappa2_max**0.5

            zfx = np.array([3.25, 0.95, 0.95])
            zfy = np.array([0.55, 3.55, 3.35])

            tau_regions = [["", "", ""], ["", "B", "A"], ["", "C", "D"]]
            tau_layer = ["", "s", "c"]
            n2clad = [0, sol_probe._n2list[0], sol_probe._n2list[-1]]
            ioffset = 5

            for i in range(0, 3):

                axs[i+ioffset].axline((0, 0), slope=0, linestyle="--", linewidth=1, c="k")
                axs[i+ioffset].axline((0, 0), (0,1), linestyle="--", linewidth=1, c="k")

                axs[i+ioffset].set_xlim([-zfx[i], zfx[i]])
                axs[i+ioffset].set_ylim([-zfy[i], zfy[i]])

                # write labels for tau regions
                if i==0:
                    axs[i+ioffset].text(0.03, 0.08, "$(\\kappa', \\kappa'')$", usetex=True, transform=axs[i+ioffset].transAxes, fontsize=fontsize*1.25, ha="left", va="center")
                if i>0:
                    axs[i+ioffset].text(0.03, 0.08, "$(\\tau_{\mathrm "+tau_layer[i]+"}', \\tau_{\mathrm "+tau_layer[i]+"}'')$", usetex=True, transform=axs[i+ioffset].transAxes, fontsize=fontsize*1.25, ha="left", va="center")
                    for iSx in [+1, -1]:
                        for itau_im in [+1, -1]:
                            sign_Sx = int(iSx*np.sign(n2clad[i].real))
                            axs[i+ioffset].text(
                                0.25*zfx[i]*iSx, 0.25*zfy[i]*itau_im,
                                "$\\mathrm{"+tau_regions[(i*2-3)*itau_im][(i*2-3)*sign_Sx]+"}_\mathrm{"+tau_layer[i]+"}$",
                                color="k", ha="center", va="center", fontsize=fontsize*1.25,
                                )

                    # draw a line where S_x = 0
                    c = "C"+str(int((-np.sign(n2clad[i].real)+1)/2)) # dielectric: C0; metal: C1
                    if np.abs(n2clad[i].real)<1e-12:
                        xSx0 = 0
                    else:
                        xSx0 = -n2clad[i].imag/n2clad[i].real

                    axs[i+ioffset].axline((0, 0), (xSx0, 1), linestyle="--", linewidth=1, c=c)
                    axs[i+ioffset].text(xSx0*zfy[i], zfy[i]*1.1,
                        "$\\bar{S}_x^{\\mathrm{(TM)}}=0$",
                        color=c, ha="center", va="center", fontsize=fontsize)

    #%% annotations for the figure
    for i in range(0, 4):

        axs[i+1].text(0.25, 1.2, "\\textbf{---} $\\bar{S}_y$", usetex=True, transform=axs[i+1].transAxes, fontsize=fontsize*1.25, color="C2", ha="center", va="center", fontweight="bold")
        axs[i+1].text(0.75, 1.2, "$\\cdot\\cdot\\cdot\\bar{S}_x$", usetex=True, transform=axs[i+1].transAxes, fontsize=fontsize*1.25, color="C4", ha="center", va="center", fontweight="bold")
        axs[i+1].text(-0.10, 0.5, "$h_z$", usetex=True, transform=axs[i+1].transAxes, fontsize=fontsize, color="C"+str(i), ha="center", va="center", rotation=90)
        axs[i+1].text(0.5, -0.40, "x (nm)", usetex=True, transform=axs[i+1].transAxes, fontsize=fontsize*1.25, ha="center", va="top")

        axs[i+1].text(0.55, 0.15, "$\\mathrm{TM_"+str(i)+"}$", usetex=True, transform=axs[i+1].transAxes, fontsize=fontsize, ha="center", va="center", fontweight="bold", color=c_hz[i])

    axSpectra.text(0.6, -0.15, "$\\beta'$", usetex=True, transform=axSpectra.transAxes, fontsize=fontsize*1.25, ha="center", va="top")
    axSpectra.set_ylabel("$\\frac{\omega}{\omega_\mathrm{p}}$", rotation=0, fontsize=fontsize*1.25)
    axSpectra.set_ylim([0.09, 0.51])
    axSpectra.set_yticks([0.1, 0.3, 0.5])
    axSpectra.set_xticks([0, 30, 60])
    axSpectra.set_xlim([0, 55])

    # continue
    sys.exit(0)

    #%% save the figure
    figname = wg["name"]
    plt.savefig(figname + '.png',format='png', dpi=300)
    # plt.savefig(figname + '.pdf',format='pdf')
    # from PIL import Image
    # Image.open(figname+'.png').convert('L').save(figname+'-bw.png')
