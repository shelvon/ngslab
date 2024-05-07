#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shelvon
@email: xiaorun.zang@outlook.com

"""

import os
import sys
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
model.ld0_target *= 1e6

geom_tFree = model.ld0_target*0.2
geom_tSub = model.ld0_target*0.2
geom_tPML = model.ld0_target*0.2

# geom_hMax = model.ld0_target/31; geom_nPML = 13; # extra fine mesh
# geom_hMax = model.ld0_target/31; geom_nPML = 27; # extra fine PML mesh
# geom_hMax = model.ld0_target/17; geom_nPML = 13; # moderate resolution
geom_hMax = model.ld0_target/31; geom_nPML = 1; # TBC debugging

geom_delPML = geom_tPML/geom_nPML
print("hMax="+f"{geom_hMax/model.ld0_target:.4f}*ld0")
print("hPML="+f"{geom_delPML/model.ld0_target:.4f}*ld0")

#%% loop in the slab thickness
tSlabArray = np.arange(500e-9, 501e-9, 20e-9)

# Here, a change of unit to micrometer is needed as well.
tSlabArray *= 1e6
for it, geom_tSlab in zip(range(tSlabArray.size), tSlabArray):
    print("tSlab["+str(it)+"] = "+f"{geom_tSlab*1e3:.0f} nm")

    # choose a waveguide
    # wg = wg_4layer(geom_tSlab)
    wg = wg_4layer_lossy(geom_tSlab)
    # wg = wg_ARROW()

    # update the waveguide geometry
    model.geom.Update(wg["intervals"], wg["nnodes"], wg["labels"])

    #-- create the mesh
    model.mesh.Create()
    # model.mesh.Plot(); #sys.exit(0);

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

    # model.mesh.Plot(); # sys.exit(0);

    model.TBC_PEP = True

    # sys.exit(0)
    #%% model build
    #---- build the model (create function space, set up materials, enable pml materials)
    # bspl_order: the order of the ngsolve.BSpline representations for dispersive and tabulated materials
    model.Build(fes_order=2, bspl_order=2, ld_scale=1e6) # bspl_order = 1 (linear), 2 (quadratic) curve

    # debug PML
    # model.SetPML(model.material.pml, pml_plot=True);

    #%% model simulation

    #---- Sweep normalized frequency and solve the problem
    model.wArray = np.array([1.0])

    # model.wArray = np.concatenate( (model.wArray, np.arange(0.6, 2.8, 0.1)) )

    model.wArray = np.unique(np.round(model.wArray*1000))/1000 # keep unique floating numbers of 3 decimal precision.

    # figure properties
    plt.close("all")

    inch2cm = 1.0/2.54  # inches-->centimeters
    figsize = 1.0*np.array([8.6, 6.4])*inch2cm
    ms = 4
    fontsize = 8
    plt.rc('font', size=fontsize)

    fig, axsSpectra = plt.subplots(figsize=figsize, ncols=2, nrows=1, constrained_layout=True, sharex=True, sharey=True);
    # fig, axsSpectra = plt.subplots(figsize=figsize, ncols=1, nrows=1, constrained_layout=True, sharex=True, sharey=True);
    ll_plotted = False

    # print("ldArray="+str(model.ld0_target/model.wArray)+" um")
    for iw in range(model.wArray.size):
    # for iw in range(model.wArray.size)[::-1]: # in reverse order
        model.w = model.wArray[iw]

        sol = model.Solve(show_pattern=False)

        #%% investigate the eigenvalues
        cmaps_r = ["C7", "C3", "C2", "C5"] # complementary color of each
        c_fem = ["g", "C7"]
        alpha_fem = [1.0, 0.6]
        c_tmatrix = ["C3"]

        markerstyle_tmatrix = dict(marker="x", s=ms*2, linewidths=0.5, alpha=0.5, zorder=1e7, clip_on=True)
        markerstyle_fem = dict(marker="o", s=round(ms*1.5), edgecolor="none", zorder=1e6, clip_on=True)

        n2list = np.asarray(model.sol._n2list)
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
        # for mode_type in [0, 1]:

            ## results from FEM simulations
            nvalid = model.sol._nvalid[mode_type]
            kappa_fem = model.sol.kappa[mode_type, :nvalid]
            kappa2_fem = model.sol.kappa2[mode_type, :nvalid]
            taus_fem = model.sol.taus[mode_type, :nvalid]
            tauc_fem = model.sol.tauc[mode_type, :nvalid]

            idx_kappa_fem = np.where(
                ~np.isnan(kappa_fem)
                & (np.abs(kappa_fem.real/kappa_fem.imag)>Q_threshold)
                & (np.abs(kappa_fem)**2<kappa2_limit)
                )[0]

            # divided tau plane into regions by the sign of Sx
            if mode_type==0: # TM mode
                Sx_s = (taus_fem.real*n2s.real+taus_fem.imag*n2s.imag)
                Sx_c = (tauc_fem.real*n2c.real+tauc_fem.imag*n2c.imag)
            elif mode_type==1: # TE mode for nonmagnetic material with (mu.real = 1, mu.imag=0)
                Sx_s = -(taus_fem.real)
                Sx_c = -(tauc_fem.real)

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
            idx = np.intersect1d(idx, idx_denser2rarer)
            idx = np.intersect1d(idx_kappa_fem, idx)
            # idx = idx_denser2rarer

            # idx = idx_kappa_fem
            idx_del = np.setdiff1d(np.arange(nvalid), idx)

            if idx.size>0:

                # modes selected by all three conditions
                c = range(idx.size)
                # c = "C0"
                axsSpectra[mode_type].scatter(np.abs(kappa_fem[idx].real)*model.k0, np.repeat(model.w, idx.size), c=c, cmap="seismic", s=8, alpha=np.exp(-np.abs(kappa2_fem[idx].imag)-kappa_tol)**0)

                # modes selected by only the condition on kappa
                # axsSpectra[mode_type].scatter(np.abs(kappa_fem[mode_type, idx_kappa_fem].real)*model.k0, np.repeat(model.w, idx_kappa_fem.size), c="C7", s=4, alpha=np.exp(-np.abs(kappa2_fem[mode_type,idx].imag)-kappa_tol)**0)
                # mode w/o any selection
                # axsSpectra[mode_type].scatter(kappa_fem[mode_type, idx_del].real*model.k0, np.repeat(model.w, idx_del.size), c="C7", s=4)

                ## complex refractive index of a dispersive medium
                # axsSpectra[0].scatter(nlist[1].real*2*np.pi/model.ld0, model.w, c="b")
                # axsSpectra[0].scatter(nlist[1].imag*2*np.pi/model.ld0, model.w, c="g")

            # axsSpectra.set_xlim([0, 250])
            axsSpectra[mode_type].set_xlim([0, 250])

        # plot two light lines once
        if not ll_plotted:

            axsSpectra[0].axline((0,0), (nc*2*np.pi/model.ld0Array[0], model.wArray[0]), c="k")
            axsSpectra[1].axline((0,0), (nc*2*np.pi/model.ld0Array[0], model.wArray[0]), c="k")

            axsSpectra[0].axline((0,0), (ns*2*np.pi/model.ld0Array[0], model.wArray[0]), c="gray")
            axsSpectra[1].axline((0,0), (ns*2*np.pi/model.ld0Array[0], model.wArray[0]), c="gray")

            ll_plotted = True

        #### at a single frequency
        # investigate eigenvalues in the complex planes
        if (model.w==1.0):

            sol_probe = copy.deepcopy(sol)

            #### results from the t-matrix calculations
            if wg["name"]=="demo_4layer":
                neff_tmatrix = np.genfromtxt("4layer_tmatrix.txt", dtype=np.complex128)
                neff_tmatrix = neff_tmatrix.reshape((2, int(neff_tmatrix.size/2)))
            elif wg["name"]=="demo_4layer_lossy":
                neff_tmatrix = np.genfromtxt("4layer_lossy_tmatrix.txt", dtype=np.complex128)
                neff_tmatrix = neff_tmatrix.reshape((2, int(neff_tmatrix.size/2)))
            else:
                neff_tmatrix = np.full((2,1), np.nan)

            nmodes_plot = np.min([12, np.size(neff_tmatrix, 1)])
            kappa_tmatrix = neff_tmatrix[:,:nmodes_plot]
            kappa2_tmatrix = kappa_tmatrix**2

            # plus a negligible imag to get the correct sign when it is pure real
            tau2_s_tmatrix = np.array(n2list[0]-kappa2_tmatrix) - 1j*1e-32
            tau2_c_tmatrix = np.array(n2list[-1]-kappa2_tmatrix) - 1j*1e-32

            taus_tmatrix = -np.sqrt(tau2_s_tmatrix)*np.sign(np.real(tau2_s_tmatrix))
            tauc_tmatrix = np.sqrt(tau2_c_tmatrix)*np.sign(np.real(tau2_c_tmatrix))

            eig_tmatrix = 0.5*(tauc_tmatrix - taus_tmatrix)


            plt.close("all")

            #### plot three regions in the complex planes of kappa, lambda, taus, and tauc
            fig, _axs = plt.subplots(figsize = figsize, ncols = 2, nrows = 2, constrained_layout = False);
            axs = _axs.flatten()

            slab.plotRegions(axs, nklist=nlist)
            # slab.plotRegions(axs, nklist=nlist, kappa_modes=kappa_tmatrix[mode_type,:])

            cmap_fem = "viridis_r"
            cmap_fem_del = "Grays"

            axsScatter = []
            for ax, zz, zz_tmatrix in zip([axs[0], axs[1], axs[2], axs[3]],
                              [sol_probe.kappa[mode_type, :sol_probe._nvalid[mode_type]],
                               sol_probe._eigval[mode_type, :sol_probe._nvalid[mode_type]],
                               sol_probe.taus[mode_type, :sol_probe._nvalid[mode_type]],
                               sol_probe.tauc[mode_type, :sol_probe._nvalid[mode_type]]],
                              [kappa_tmatrix[mode_type,:],
                               eig_tmatrix[mode_type,:],
                               taus_tmatrix[mode_type,:],
                               tauc_tmatrix[mode_type,:]]
                              ):

                ax.scatter(zz_tmatrix.real, zz_tmatrix.imag, c=c_tmatrix[0], **markerstyle_tmatrix)

                ax.scatter(zz[idx_del].real, zz[idx_del].imag, c=c_fem[1], alpha=alpha_fem[1], **markerstyle_fem)
                # markerstyle_fem["zorder"] += 1
                axsScatter.append(ax.scatter(zz[idx].real, zz[idx].imag, c=range(idx.size)[::-1], alpha=alpha_fem[0], **markerstyle_fem))



            # extract the field profile of selected modes
            axs_names = ["kappa", "eigval", "taus", "tauc"]
            axs_select = np.asarray(axsScatter)

            # plt.close(1) # close the figure for plotting the dispersion curve
            ####  interactively select a mode by clicking at the nearest point in the complex plane
            mode_selector = slab.SlabWaveGuide.modeSelector(
                axs_select, sol_probe, axs_names, mode_type=mode_type, nmodes = 2,
                )

            kappa_max = np.max(np.abs(np.asarray(sol_probe._nlist)))
            #### plot limits for the 4layer, lossless waveguide case
            if wg["name"]=="demo_4layer" or wg["name"]=="demo_4layer_lossy":
                axs[0].set_xlim([0.21, kappa_max*1.02])
                axs[0].set_ylim([-0.02, 0.67])

                axs[0].text(sol_probe._nlist[0].real+0.03, 0.32, '$n_{\mathrm{s}}$', usetex=True, fontsize=fontsize*1.25, ha="left", va="center", fontweight="book")
                axs[0].text(sol_probe._nlist[-1].real+0.03, 0.32, '$n_{\mathrm{c}}$', usetex=True, fontsize=fontsize*1.25, ha="left", va="center", fontweight="book")
                # inset showing a zoom in area
                # if wg["name"]=="demo_4layer_lossy":
                # axs[0].set_ylim([-0.02, 0.67])
                x1, x2, y1, y2 = sol_probe._nlist[0].real*0.99, kappa_max*0.99, -0.0001, 0.0003  # subregion of the original image
                axins = axs[0].inset_axes(
                    [0.3, 0.75, 0.8, 0.35],
                    xlim=(x1, x2), ylim=(y1, y2), xticks=[], yticks=[])
                zz = sol_probe.kappa
                zz_tmatrix = kappa_tmatrix
                axins.scatter(zz[mode_type,idx_del].real, zz[mode_type,idx_del].imag, c=c_fem[1], alpha=alpha_fem[1], **markerstyle_fem)
                markerstyle_fem["zorder"] += 1
                axins.scatter(zz[mode_type,idx].real, zz[mode_type,idx].imag, c=range(idx.size), cmap=cmap_fem, alpha=alpha_fem[0], **markerstyle_fem)
                axins.scatter(zz_tmatrix[mode_type,:].real, zz_tmatrix[mode_type,:].imag, c=c_tmatrix[0], **markerstyle_tmatrix)
                axins.axline((0, 0), (1,0), linestyle="--", linewidth=1, c="k")

                axs[0].indicate_inset_zoom(axins, edgecolor="k")

                axs[1].set_xlim([-1.35, 1.35])
                axs[1].set_ylim([-1.05, 1.05])
                axs[2].set_xlim([-1.75, 1.75])
                axs[2].set_ylim([-0.75, 0.75])
                axs[3].set_xlim([-1.45, 1.45])
                axs[3].set_ylim([-1.75, 1.75])
                if wg["name"]=="demo_4layer_lossy":

                    axs[2].set_xlim([-0.00110, 0.00110])
                    axs[3].set_xlim([-0.00030, 0.00030])
                    axs[2].set_ylim([-0.75, 0.75])
                    axs[3].set_ylim([-1.45, 1.45])

                    axs[2].set_xscale('symlog')
                    axs[3].set_xscale('symlog')
                    axs[2].set_xticks([-0.001, 0, 0.001])
                    axs[3].set_xticks([-0.0002, 0, 0.0002])

            plt.subplots_adjust(hspace=0.5)
            plt.subplots_adjust(wspace=0.4)
    sys.exit(0)
    # continue
    #%% save the figure
    figname = wg["name"]
    plt.savefig(figname + '.png',format='png', dpi=300)
    # plt.savefig(figname + '.pdf',format='pdf')
    # from PIL import Image
    # Image.open(figname+'.png').convert('L').save(figname+'-bw.png')
