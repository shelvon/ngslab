#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shelvon
@email: xiaorun.zang@outlook.com

"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

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
# python3 -m pip install --upgrade .
#
# - Method 3, conda install the local package build by conda-build.
# conda install path/to/local/package

# import slab

# print(dir()) # print what have been imported

# sys.exit(0)
#%% 4-layer waveguide:
# a waveguide given in Fig. 2 of Ref. [J. Chilwell and I. Hodgkinson, ‘Thin-films field-transfer matrix theory of planar multilayer waveguides and reflection from prism-loaded waveguides’, J. Opt. Soc. Am. A, vol. 1, no. 7, pp. 742–753, Jul. 1984, doi: 10.1364/JOSAA.1.000742]
def wg_4layer(geom_tSlab):
    nSlab = np.maximum(geom_nPML, round(np.ceil(geom_tSlab/geom_hMax)))

    wg = dict(
        # geometry
        name = "demo_4layer",
        intervals = [-4*geom_tSlab-geom_tSub-geom_tPML, -4*geom_tSlab-geom_tSub, -4*geom_tSlab, -3*geom_tSlab, -2*geom_tSlab, -geom_tSlab, 0, geom_tFree, geom_tFree+geom_tPML],
        labels = ["pml_left", "sub", "slab4","slab3","slab2","slab1", "air", "pml_right"],
        nnodes = [geom_nPML, round(np.ceil(geom_tSub/geom_hMax)), nSlab, nSlab, nSlab, nSlab, round(np.ceil(geom_tFree/geom_hMax)), geom_nPML],
        # material
        nk = {"default":1.0, "nc": 1.0, "ns": 1.5, "n1": 1.66, "n2": 1.53, "n3": 1.60, "n4": 1.66},
        map = {"pml_left": "ns", "sub":"ns", "slab4":"n4", "slab3":"n3", "slab2":"n2", "slab1":"n1", "air":"nc", "pml_right":"nc"},
        )

    return wg

#### 4-layer lossy waveguide:
def wg_4layer_lossy(geom_tSlab):
    wg = wg_4layer(geom_tSlab);
    wg.update(
        name="demo_4layer_lossy",
        nk={"default":1.0, "nc": 1.0, "ns": 1.5, "n1": 1.66*(1+.0001j), "n2": 1.53*(1+.0001j), "n3": 1.60, "n4": 1.66},
        # nk={"default":1.0, "nc": 1+0.0j, "ns": 1.5+0.5j, "n1": 1.66*(1+0.000j), "n2": 1.53*(1+0.0000j), "n3": 1.60, "n4": 1.66},
        map = {"pml_left": "ns", "sub":"ns", "slab4":"n4", "slab3":"n3", "slab2":"n2", "slab1":"n1", "air":"nc", "pml_right":"nc"},
        )

    return wg

#### ARROW waveguide:
# in Fig. 2 of Ref. [H. P. Uranus, H. J. W. M. Hoekstra, and E. Van Groesen, ‘Simple high-order Galerkin finite element scheme for the investigation of both guided and leaky modes in anisotropic planar waveguides’, Optical and Quantum Electronics, vol. 36, no. 1, pp. 239–257, Jan. 2004, doi: 10.1023/B:OQEL.0000015643.52433.f0.]
def wg_ARROW():
    d1 = 2.0985e-6*1e6;
    d2 = 0.1019e-6*1e6;
    d3 = 4e-6*1e6;
    nSlab = np.maximum(geom_nPML, round(np.ceil(geom_tSlab/geom_hMax)));
    wg = dict(
        name = "demo_ARROW",
        intervals = [-d1-d2-d3-geom_tSub-geom_tPML, -d1-d2-d3-geom_tSub, -d1-d2-d3, -d2-d3, -d3, 0, geom_tFree, geom_tFree+geom_tPML],
        labels = ["pml_left", "sub", "slab1","slab2","slab3", "air", "pml_right"],
        nnodes = [geom_nPML, round(np.ceil(geom_tSub/geom_hMax)), nSlab, nSlab, nSlab, round(np.ceil(geom_tFree/geom_hMax)), geom_nPML],
        nk = {"default":1.0, "nc": 1.0, "ns": 3.5, "n1": 1.45, "n2": 3.5, "n3": 1.45},
        map = {"pml_left": "ns", "sub":"ns", "slab1":"n1", "slab2":"n2", "slab3":"n3", "air":"nc", "pml_right":"nc"},
        )

    return wg

#%% model creation
#---- Geometric Parameters
model = slab.SlabWaveGuide();
model.ld0_target = 6328e-10

# use micrometer may yield lower condition number
# model.ld0_target *= 1e6

geom_tPML = model.ld0_target*1.0
geom_tFree = model.ld0_target*1.5
geom_tSub = model.ld0_target*1.5

# geom_hMax = model.ld0_target/31; geom_nPML = 13; # extra fine mesh
# geom_hMax = model.ld0_target/31; geom_nPML = 27; # extra fine PML mesh
geom_hMax = model.ld0_target/17; geom_nPML = 17; # moderate resolution
# geom_hMax = model.ld0_target/17; geom_nPML = 1; # TBC test
# geom_hMax = model.ld0_target/31; geom_nPML = 1; # TBC debugging

# pml_params=[2, 1]
# pml_params=[2, 100]
# pml_params=[11, 100]
# pml_params=[101, 100]
# pml_params=[2, 10]
# pml_params=[0, 10]


pml_params=[11, 10]
# pml_params=[-1, 10]

geom_delPML = geom_tPML/geom_nPML
print("hMax="+f"{geom_hMax/model.ld0_target:.4f}*ld0")
print("hPML="+f"{geom_delPML/model.ld0_target:.4f}*ld0")

#%% loop in the slab thickness
tSlabArray = np.arange(500e-9, 501e-9, 20e-9)

plt.close("all")
# A change of unit to micrometer.
# tSlabArray *= 1e6
for it, geom_tSlab in zip(range(tSlabArray.size), tSlabArray):
    print("tSlab["+str(it)+"] = "+f"{geom_tSlab*1e3:.0f} nm")

    # choose a waveguide
    wg = wg_4layer(geom_tSlab)
    # wg = wg_4layer_lossy(geom_tSlab)
    # wg = wg_ARROW()

    # update the waveguide geometry
    model.geom.Update(wg["intervals"], wg["nnodes"], wg["labels"])

    #-- create the mesh
    model.mesh.Create()
    # model.mesh.Plot(); sys.exit(0);

    model.material.nk = wg["nk"]
    model.material.map = wg["map"]

    #%% method 1: real-time PML calculations
    # enable PML domains
    model.material.pml = {"pml_left":True, "pml_right":True}

    # model.mesh.Plot(); # sys.exit(0);

    #### model build
    #---- build the model (create function space, set up materials, enable pml materials)
    # bspl_order: the order of the ngsolve.BSpline representations for dispersive and tabulated materials
    # bspl_order = 1 (linear), 2 (quadratic) curve
    model.Build(fes_order=2, bspl_order=2, ld_scale=1, pml_params=pml_params)

    #### model simulation

    # model.w = 1.0 # by default model.w == 1.0
    t0 = time.process_time()
    sol_PML = model.Solve(show_pattern=False)
    t1 = time.process_time()
    t_CPU = t1 - t0

    print('CPU Execution time:', t_CPU, 'seconds')

    # sys.exit(0)
    n2list = np.asarray(sol_PML.n2list)
    nlist = np.asarray(n2list)**0.5
    n2delta = n2list[-1] - n2list[0]
    n2Sigma = n2list[-1] + n2list[0]
    ns, nc = nlist[[0, -1]]
    n2s, n2c = nlist[[0, -1]]**2

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

        ## results from TBC simulations
        kappa_PML = sol_PML.neff[mode_type,:]
        kappa2_PML = kappa_PML**2

        idx_kappa_PML = np.where(
            ~np.isnan(kappa_PML)
            & (np.abs(kappa_PML.real/kappa_PML.imag)>Q_threshold)
            & (np.abs(kappa2_PML)<kappa2_limit)
            )[0]

        # print(kappa_PML[idx_kappa_PML])

    # sys.exit(0)
    #%% method 2: real-time TBC calculations
    # disable PML
    model.material.pml = {"pml_left":False, "pml_right":False} # False by default

    # and then set the transparent boundary condition
    # setTBC() accepts two numbers defining the indices of the leftmost to the rightmost semi-infinite regions that bounding the core waveguide region
    model.setTBC(tbc_doms=[1, -2]) # by default: tbc_doms=[1, -2]
    # model.setTBC(tbc_doms=[0, -1]) # test

    # model.mesh.Plot(); # sys.exit(0);

    # model.TBC_PEP = False

    # sys.exit(0)

    ##### model build
    #---- build the model (create function space, set up materials, enable pml materials)
    # bspl_order: the order of the ngsolve.BSpline representations for dispersive and tabulated materials
    model.Build(fes_order=2, bspl_order=2, ld_scale=1) # bspl_order = 1 (linear), 2 (quadratic) curve

    #### model simulation

    t0 = time.process_time()
    sol_TBC = model.Solve(show_pattern=False)
    t1 = time.process_time()
    t_CPU = t1 - t0

    print('CPU Execution time:', t_CPU, 'seconds')

    # the last one will be plotted in the complex planes
    # mode_type == 0: TM mode; mode_type == 1: TE mode
    for mode_type in [0]:
    # for mode_type in [1, 0]:

        ## results from TBC simulations
        nvalid = sol_TBC._nvalid[mode_type]
        kappa_TBC = sol_TBC.kappa[mode_type, :nvalid]
        kappa2_TBC = sol_TBC.kappa2[mode_type, :nvalid]
        taus_TBC = sol_TBC.taus[mode_type, :nvalid]
        tauc_TBC = sol_TBC.tauc[mode_type, :nvalid]

        idx_kappa_TBC = np.where(
            ~np.isnan(kappa_TBC)
            & (np.abs(kappa_TBC.real/kappa_TBC.imag)>Q_threshold)
            & (np.abs(kappa2_TBC)<kappa2_limit)
            )[0]

        # divided tau plane into regions by the sign of Sx
        if mode_type==0: # TM mode
            Sx_s = (taus_TBC.real*n2s.real+taus_TBC.imag*n2s.imag)
            Sx_c = (tauc_TBC.real*n2c.real+tauc_TBC.imag*n2c.imag)
        elif mode_type==1: # TE mode for nonmagnetic material with (mu.real = 1, mu.imag=0)
            Sx_s = -(taus_TBC.real)
            Sx_c = -(tauc_TBC.real)

        # Keep modes that do not have energy feeding into the waveguide region from the infinity
        # That means, the situations of both the power flow direction points into the waveguide
        # and density increases away from the waveguide cannot be met simultaneously.
        #
        #
        #FIXME: According to the following paper
            # [1] T. Tamir and F. Kou, ‘Varieties of leaky waves and their excitation along multilayered structures’, IEEE Journal of Quantum Electronics, vol. 22, no. 4, pp. 544–551, Apr. 1986, doi: 10.1109/JQE.1986.1072991.
            # The contra-leaky mode V cannot be excited by any realistic sources.
            # However, this argument is not so obvious without giving any further explanation in that paper.
        idx_s = np.where(
            ( ~((Sx_s>-kappa_tol) & (taus_TBC.imag>-kappa_tol)) )
            )[0]
        idx_c = np.where(
            ( ~((Sx_c<kappa_tol) & (tauc_TBC.imag<kappa_tol)) )
            )[0]

        # mode indices of power flux from the denser cladding layer to the rarer cladding layer
        if ns.real>=nc.real:
            idx_denser2rarer = np.where(
                (~(
                    ((Sx_s>-kappa_tol) & (taus_TBC.imag<kappa_tol)) # power flow type As
                    &((Sx_c>-kappa_tol) & (tauc_TBC.imag<kappa_tol)) # power flow type Cc
                   ))
                )[0]

        idx = np.intersect1d(idx_s, idx_c)
        idx = np.intersect1d(idx, idx_denser2rarer)
        idx = np.intersect1d(idx_kappa_TBC, idx)
        # idx = idx_denser2rarer

        # idx = idx_kappa_fem
        idx_del = np.setdiff1d(np.arange(nvalid), idx)

    #%% method3: load t-matrix calculations in the literature
        if wg["name"]=="demo_4layer":
            neff_tmatrix = np.genfromtxt("4layer_tmatrix.txt", dtype=np.complex128)
            neff_tmatrix = neff_tmatrix.reshape((2, int(neff_tmatrix.size/2)))
        elif wg["name"]=="demo_4layer_lossy":
            neff_tmatrix = np.genfromtxt("4layer_lossy_tmatrix.txt", dtype=np.complex128)
            neff_tmatrix = neff_tmatrix.reshape((2, int(neff_tmatrix.size/2)))
        else:
            neff_tmatrix = np.full((2,1), np.nan)

        nmodes_plot = np.min([12, np.size(neff_tmatrix, 1)])
        kappa_tmatrix = neff_tmatrix[mode_type,:nmodes_plot]
        kappa2_tmatrix = kappa_tmatrix**2

        # plus a negligible imag to get the correct sign when it is pure real
        tau2_s_tmatrix = np.array(n2list[0]-kappa2_tmatrix) - 1j*1e-32
        tau2_c_tmatrix = np.array(n2list[-1]-kappa2_tmatrix) - 1j*1e-32

        taus_tmatrix = -np.sqrt(tau2_s_tmatrix)*np.sign(np.real(tau2_s_tmatrix))
        tauc_tmatrix = np.sqrt(tau2_c_tmatrix)*np.sign(np.real(tau2_c_tmatrix))

        eig_tmatrix = 0.5*(tauc_tmatrix - taus_tmatrix)


    #%% plotting the comparison
    # figure properties
    # plt.close("all")

    inch2cm = 1.0/2.54  # inches-->centimeters
    figsize = 1.0*np.array([8.6, 9.4])*inch2cm
    ms = 8
    fontsize = 8
    plt.rc('font', size=fontsize)

    fig, axs = plt.subplots(figsize=figsize, ncols=1, nrows=2, constrained_layout=True, sharex=True, sharey=False, clip_on=True);
    ll_plotted = False

    # sys.exit(0)
    #### investigate the eigenvalues
    cmaps_r = ["C7", "C3", "C2", "C5"] # complementary color of each
    c_fem = ["g", "C7"]
    alpha_fem = [1.0, 0.6]
    c_tmatrix = ["C3"]

    markerstyle_tmatrix = dict(marker="x", s=ms*4, linewidths=1.0, alpha=1.0, zorder=1e7, clip_on=True)
    markerstyle_fem = dict(marker="o", s=round(ms*1.5), edgecolor="none", zorder=1e6, clip_on=True)

    kappa_PML_selected = kappa_PML[idx_kappa_PML]
    kappa_TBC_selected = kappa_TBC[idx_kappa_TBC]
    idx_TBC_closest, idx_PML_closest = [], []
    for kappa_this in kappa_tmatrix:
        idx_TBC_closest.append(np.abs(kappa_TBC_selected-kappa_this).argmin())
        idx_PML_closest.append(np.abs(kappa_PML_selected-kappa_this).argmin())
        # print(idx_TBC_closest[-1], idx_PML_closest[-1])

    yerr_TBC = (kappa_tmatrix.imag-kappa_TBC_selected[idx_TBC_closest].imag)/np.abs(kappa_tmatrix)
    xerr_TBC = (kappa_tmatrix.real-kappa_TBC_selected[idx_TBC_closest].real)/np.abs(kappa_tmatrix)

    yerr_PML = (kappa_tmatrix.imag-kappa_PML_selected[idx_PML_closest].imag)/np.abs(kappa_tmatrix)
    xerr_PML = (kappa_tmatrix.real-kappa_PML_selected[idx_PML_closest].real)/np.abs(kappa_tmatrix)
    x, y = np.arange(kappa_tmatrix.size), np.full(np.shape(kappa_tmatrix), 0)
    x = kappa_tmatrix
    # compare modes by PML-FEM and TBC-FEM
    if idx.size>0:
        # axs[0].errorbar(
        #     x, y,
        #     yerr = 1/(-np.log10(np.abs(yerr_TBC)))**2,
        #     # xerr = 1/(-np.log10(np.abs(xerr_TBC)))**2,
        #     fmt="none", ecolor = "C3",
        #     capsize=4,
        #     )
        # axs[0].errorbar(
        #     x, y,
        #     yerr = 1/(-np.log10(np.abs(yerr_PML)))**2,
        #     # xerr = 1/(-np.log10(np.abs(xerr_PML)))**2,
        #     fmt="none", ecolor = "C7",
        #     capsize=2,
        #     )

        axs[0].plot(x, (np.abs(xerr_TBC)), 'C3-o', ms=3, lw=1.0, label="|Re($\\kappa_{\mathrm{TBC}}-\\kappa_{\mathrm{ref}}$)|")
        axs[0].plot(x, (np.abs(yerr_TBC)), 'C3--<', ms=3, lw=1.0, label="|Im($\\kappa_{\mathrm{TBC}}-\\kappa_{\mathrm{ref}}$)|")
        axs[0].plot(x, (np.abs(xerr_PML)), 'C2-o', ms=3, lw=1.0, label="|Re($\\kappa_{\mathrm{PML}}-\\kappa_{\mathrm{ref}}$)|")
        axs[0].plot(x, (np.abs(yerr_PML)), 'C2--<', ms=3, lw=1.0, label="|Im($\\kappa_{\mathrm{PML}}-\\kappa_{\mathrm{ref}}$)|")

        axs[0].set_yscale('log')
        axs[0].axvline(nc, ls="--", c="k")
        axs[0].axvline(ns, ls="--", c="k")

        axs[0].legend(handlelength=2)
    # sys.exit(0)
    # compare modes by PML-FEM and TBC-FEM
    if idx.size>0:
        axs[1].scatter(kappa_tmatrix.real, kappa_tmatrix.imag, c="C7", marker="x", s=ms*4, linewidths=2.0, alpha=1.0, clip_on=True, label="T-matrix")
        axs[1].scatter(
            np.abs(kappa_PML[idx_kappa_PML].real), kappa_PML[idx_kappa_PML].imag,
            s=ms*2,
            c="C2",
            marker="<",
            alpha=np.exp(-np.abs(kappa2_PML[idx_kappa_PML].imag)-kappa_tol)**0,
            label="PML-FEM",
            )

        axs[1].scatter(
            np.abs(kappa_TBC[idx].real), kappa_TBC[idx].imag,
            s=ms,
            # c=range(idx.size)[::-1],
            c="C3",
            # cmap="inferno",
            # alpha=np.exp(-np.abs(kappa2_TBC[idx].imag)-kappa_tol)**0,
            label="TBC-FEM",
            )
        axs[1].legend(handlelength=4, )
        axs[1].axvline(nc, ls="--", c="k")
        axs[1].axvline(ns, ls="--", c="k")

    axs[1].set_xlim([0.3, 1.65])
    axs[1].set_xlabel("$\\kappa '$")
    axs[1].set_ylabel("$\\kappa ''$")

    axs[0].set_ylim([1e-18, 1])
    axs[1].set_ylim([-0.05, 0.45])

    # axs[0].set_xlabel("$\\kappa '$")
    # axs[0].set_ylabel("$|\\Delta \\kappa| = |\\kappa - \\kappa_{\mathrm{ref}}|$")
    axs[0].set_ylabel("relative difference")

    for i in range(2):
        axs[i].text(-0.05, 1.05, '('+chr(i+97)+')', transform=axs[i].transAxes, fontsize=fontsize*1.25, ha="right", va="bottom", fontweight="book")


    axs[0].set_title("$s' + i s'' = (1 "+f"{pml_params[0]-1:+G}"+"|\\frac{\\Delta x}{L}|^2) + "+str(pml_params[1])+"i |\\frac{\\Delta x}{L}|^2$")


    plt.legend(framealpha=1.0)
    # sys.exit(0)
    # continue
    #%% save the figure
    # figname = wg["name"]+"pml_1"
    # figname = wg["name"]+"pml_2"
    figname = wg["name"]+"pml_params_"+str(pml_params[0]-1)+"_"+str(pml_params[1])
    # plt.savefig(figname + '.png',format='png', dpi=300)
    plt.savefig(figname + '.pdf',format='pdf')
    # from PIL import Image
    # Image.open(figname+'.png').convert('L').save(figname+'-bw.png')
