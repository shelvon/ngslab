#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shelvon
@email: xiaorun.zang@outlook.com

"""

import ngsolve as ngs

import sys
import copy

import slepc4py
slepc4py.init(sys.argv)

from slepc4py import SLEPc
# slepc4py.SLEPc.ComplexType
# This returns datatype "numpy.complex128"

import numpy as np

# physical constants
from . import const as _const
from .mat import ngs2numpyMat, ngs2petscMatAIJ, petscMat2numpyMat
from .plot import _fig, _plt

class SlabWaveGuide:

    class Geometry:

        def __init__(self, intervals=(0, 1), nnodes=(17, 0), labels=("freespace", "dummy")):
            self.intervals = intervals
            self.labels = labels
            self.nnodes = nnodes

        def Update(self, intervals, nnodes, labels):
            self.intervals = intervals
            self.labels = labels
            self.nnodes = nnodes

            # By default,
            # the physical region excludes the left and right outmost regions, which are assumed to be PMLs,
            self.region_phy = [self.intervals[1], self.intervals[-2]]
            # the waveguide region further excludes the left and right substrate and superstrate regions.
            self.region_wg = [self.intervals[2], self.intervals[-3]]

    class Mesh:
        import netgen # installed together with ngsolve

        def Plot(self, _ax=None):
            if _ax == None:
                self._fig, self._ax = _plt.subplots(figsize =   (1.5*_fig.h*_fig.inch2cm*_fig.zoom, 0.35*_fig.h*_fig.inch2cm*_fig.zoom), constrained_layout = True);
            else:
                self._ax = _ax

            x_v = [p[0] for p in self._ngsolve_mesh.ngmesh.Points()]
            self._ax.plot(self._obj_geom.intervals, [0.0 for bnds in self._obj_geom.intervals], 'k-', linewidth=1, marker='|', markersize=12)
            self._ax.plot(x_v, [0.0 for v in x_v], 'C0-', marker='o', markersize=4)

            ab = [self._obj_geom.intervals[0], self._obj_geom.intervals[-1]]
            mid_points = (0.5*(np.asarray(self._obj_geom.intervals[:-1]) + np.asarray(self._obj_geom.intervals[1:])) \
                - ab[0] ) / (ab[1]-ab[0])
            for ip, mp in zip(range(mid_points.size), mid_points):
                self._ax.text(mp.real, 0.3, self._obj_geom.labels[ip], transform=self._ax.transAxes, ha="center")

            # self._ax.set_ylim(-0.1, 0.1)
            self._ax.set_xlim(x_v[0], x_v[-1])
            self._ax.set_xlabel("x")

        def Create(self, bnd_left=0, bnd_right=-1):

            self._netgen_mesh = self.netgen.meshing.Mesh(dim=1)
            self.pids = []

            self._AddIntervals()

            self._bnd_left = "left"
            self._bnd_right = "right"
            idx_left = self._netgen_mesh.AddRegion(self._bnd_left, dim=0)
            idx_right = self._netgen_mesh.AddRegion(self._bnd_right, dim=0)
            self._netgen_mesh.Add (self.netgen.meshing.Element0D (self.pids[bnd_left], index=idx_left))
            self._netgen_mesh.Add (self.netgen.meshing.Element0D (self.pids[bnd_right], index=idx_right))

            self.npoints = len(self.pids)

            self.points = self._netgen_mesh.Points()

            self.px = [0.0]*self.npoints
            self.py = [0.0]*self.npoints
            self.pz = [0.0]*self.npoints
            for ip in range(self.npoints):
                self.px[ip], self.py[ip], self.pz[ip] = self.points[ip+1].p

            self.cpoint = [sum(self.px)/self.npoints, sum(self.py)/self.npoints, sum(self.pz)/self.npoints]

            self._ngsolve_mesh = ngs.Mesh(self._netgen_mesh)

        def _AddIntervals(self):
            noLabel = False

            if self._obj_geom.labels == None:
                noLabel = True

            self.pids.append (self._netgen_mesh.Add (self.netgen.meshing.MeshPoint (self.netgen.meshing.Pnt(self._obj_geom.intervals[0], 0, 0))))

            nsegments = len(self._obj_geom.intervals)-1
            ig = 0 # an offset of the point index for each segment in the case of multiple segments

            for j in range(nsegments):
                interval = (self._obj_geom.intervals[j], self._obj_geom.intervals[j+1])
                nnode = self._obj_geom.nnodes[j]
                label = None
                if not noLabel==True:
                    label = self._obj_geom.labels[j]

                idx = self._netgen_mesh.AddRegion(label, dim=1)
                for i in range(1, nnode+1):
                    self.pids.append (self._netgen_mesh.Add (self.netgen.meshing.MeshPoint (self.netgen.meshing.Pnt(interval[0]+i/nnode*(interval[1]-interval[0]), 0, 0))))
                    # print(idx, ig, i, pids[ig+i-1], pids[ig+i])
                    self._netgen_mesh.Add (self.netgen.meshing.Element1D ([self.pids[ig+i-1], self.pids[ig+i]], index=idx))

                ig += nnode

        def __init__(self, obj_geom):

            self._obj_geom = obj_geom

    class Material:
        def __init__(self, obj_model, nk={"default":1}, mu={"default":1}):
            self.nk = nk
            self.mu = mu
            self._tabulated = []
            self.map = {}
            self._created = False
            self._model = obj_model

        # dispersive (i.e., frequency-dependent) materials
        def refLoad(self, label="agjc"):
            loaded = False

            for il in range(len(self._tabulated)):
                label_created = self._tabulated[il][0]
                if label_created==label:
                    loaded = True

            if not loaded:
                refs = np.genfromtxt(label+".ref")
                wls = refs[:,0]
                nks_re = refs[:,1]
                nks_im = refs[:,2]
                nks = nks_re + 1j*nks_im

                self._tabulated.append([label, wls, nks])

        def refPlot(self, label="agjc"):
            found = False
            for il in range(len(self._tabulated)):
                label_created = self._tabulated[il][0]
                if label_created==label:
                    found = True
                    break
            if found:
                self._fig, self._ax = _plt.subplots(figsize = (0.86*_fig.h*_fig.inch2cm*_fig.zoom, _fig.h*_fig.inch2cm*_fig.zoom), constrained_layout = True);

                n=self._tabulated[il][2]
                # epsilon=n**2
                # wl=self._tabulated[il][1]*1e6

                _plt.plot(n.real, n.imag, '-o', lw=1, label='n');
                _plt.legend()

                _plt.title("Loaded from "+label+".ref")
                _plt.xlabel("n (-), k (--), epsilon'(-o), epsilon''(--o)")
                _plt.ylabel("$\\lambda (\mathrm {um})$", fontsize=_fig.fs*1.25)
            else:
                print('call .refLoad("'+label+'") to load the _tabulated refractive indices first')

        # resolve the nk or mu with a given label to the exact value
        def _resolve(self, property, label):
            val = 1.0
            resolved = False
            if property=="epsilon":
                try: # Check if the labelled material in the list
                    val = self.nk[label]

                    if type(val) is str:
                        # Is it necessary to have recursive string chain? Does not make sense.
                        # val = self._resolve("nk", self.nk[label]) # search recursively

                        for il in range(len(self._ngsolve_BSplines)):
                            label_created = self._ngsolve_BSplines[il][0]
                            if label_created==val:
                                val = self._ngsolve_BSplines[il][2]
                                resolved = True
                                break

                    # coordinate-dependent index profile must be of
                    # type "ngs.fem.CoefficientFunction"
                    elif type(val) is ngs.fem.CoefficientFunction:
                        resolved = True
                    else:
                        resolved = True

                except:
                    pass
                    # print("Cannot find nk values of "+label+"!")

                # convert from the refractive index to the epsilon value
                val = val**2

            if property=="mu":
                try: # Check if the labelled material in the list
                    val = self.mu[label]

                    if type(val) is str:

                        for il in range(len(self._ngsolve_BSplines)):
                            label_created = self._ngsolve_BSplines[il][0]
                            if label_created==val:
                                val = self._ngsolve_BSplines[il][2]
                                resolved = True
                                break
                    elif type(val) is ngs.fem.CoefficientFunction:
                        resolved = True
                    else:
                        resolved = True

                except:
                    pass
                    # print("Cannot find mu values of "+label+"!")

            return val, resolved

        def Create(self, bspl_order=1, ld_scale=1):
            self._bspl_order = bspl_order # using piecewise polynomial function of order (bspl_order-1)
            self._ngsolve_BSplines = []

            if not self._created:
                #---- Load dispersive material properties into self._tabulated list
                # dispersive epsilon (relative permittivity)
                for label in self.nk:
                    if type(self.nk[label]) is str:
                        self.refLoad(self.nk[label])

                # dispersive mu (relative permeability)
                for label in self.mu:
                    if type(self.mu[label]) is str:
                        self.refLoad(self.mu[label])

                #---- Create ngs.BSpline representations for dispersive and tabulated materials
                for il in range(len(self._tabulated)):
                    label = self._tabulated[il][0]
                    ldTable = self._tabulated[il][1]*ld_scale
                    nTable = self._tabulated[il][2][::-1]
                    wTable = self._model.ld0_target/ldTable[::-1]

                    ngsolve_bspline = ngs.BSpline(self._bspl_order, wTable.tolist(), nTable.real.tolist())(self._model._ngsolve_w) + 1j*ngs.BSpline(bspl_order, wTable.tolist(), nTable.imag.tolist())(self._model._ngsolve_w)
                    # ngsolve_bspline = ngs.sin(self._model._ngsolve_w) + 1j*ngs.sin(2*self._model._ngsolve_w) # test: Sin is ngs.sin, a builtin CoefficientFunction

                    self._ngsolve_BSplines.append([label, wTable, ngsolve_bspline]) # nk
                    # self._ngsolve_BSplines.append([label, wTable, ngsolve_bspline**2]) # epsilon

                self._created = True

            #---- Set up coefficient functions for epsilon and mu
            try:
                self.default_epsilon = self.nk["default"]**2
            except:
                self.default_epsilon = 1.0
            try:
                self.default_mu = self.mu["default"]**2
            except:
                self.default_mu = 1.0

            self._map_epsilon = dict.fromkeys(self._model.geom.labels, self.default_epsilon)
            self._map_mu = dict.fromkeys(self._model.geom.labels, self.default_mu)

            # domain-wise scaling function, 1 for non-PML domains
            self._map_sx = dict.fromkeys(self._model.geom.labels, 1.0)

            # Loop all geometry domains
            for idom in range(len(self._model.geom.labels)):
                label_dom = self._model.geom.labels[idom]

                # search if the domain is assigned by a material
                for label_material in self.map:
                    if label_material==label_dom:
                        val, resolved = self._resolve("epsilon", self.map[label_material])
                        if resolved:
                            self._map_epsilon[label_dom] = val
                        val, resolved = self._resolve("mu", self.map[label_material])
                        if resolved:
                            self._map_mu[label_dom] = val
                        break

            # print("map_epsilon = ", self._map_epsilon);
            # print("map_mu = ", self._map_mu);

        def _BSplinePlot(self, label="agjc", ax=None):
            _ngsolve_w_raw = self._model._ngsolve_w.Get()
            found = False
            for il in range(len(self._ngsolve_BSplines)):
                label_created = self._ngsolve_BSplines[il][0]
                if label_created==label:
                    found = True
                    break
            if found:
                if ax==None:
                    fig, ax = _plt.subplots(figsize = (0.86*_fig.h*_fig.inch2cm*_fig.zoom, _fig.h*_fig.inch2cm*_fig.zoom), constrained_layout = True);

                wArray_plot = self._ngsolve_BSplines[il][1]
                nk_bspl = self._ngsolve_BSplines[il][2]
                epsilon_bspl = nk_bspl**2
                nk_plot = np.full_like(wArray_plot, np.nan, dtype=np.complex128)
                epsilon_plot = np.full_like(wArray_plot, np.nan, dtype=np.complex128)

                for iw in range(wArray_plot.size):
                    self._model._ngsolve_w.Set(wArray_plot[iw])
                    mesh_pt = self._model.mesh._ngsolve_mesh(self._model.mesh.cpoint)
                    nk_plot[iw] = (nk_bspl(mesh_pt)[0])[0]
                    epsilon_plot[iw] = (epsilon_bspl(mesh_pt)[0])[0]

                ax.legend();

                kArray_plot = 2*np.pi/self._model.ld0_target*wArray_plot
                ax.plot(nk_plot[:-2].real*kArray_plot[:-2], wArray_plot[:-2], 'C0-', label="n")
                ax.plot(nk_plot[:-2].imag*kArray_plot[:-2], wArray_plot[:-2], 'C0-.', label="k")

                ax.plot(1*kArray_plot[:-2], wArray_plot[:-2], 'k-', label="light line")
                ax.plot(1.5*kArray_plot[:-2], wArray_plot[:-2], 'k-', label="light line")
                ax.set_ylim([0, wArray_plot[-2]])

            self._model._ngsolve_w.Set(_ngsolve_w_raw)

    # Values of some coefficients are taken from the paper below.
    # "S. D. Gedney, Introduction to the Finite-Difference Time-Domain (FDTD) Method for Electromagnetics (Springer International Publishing, Cham, 2011)"
    # Here, the PML material method is implemented differently from the function ngs.Mesh.SetPML, which is actually a complex coordinate transform.
    def SetPML(self, pml_flags, pml_plot=False, log_plot=False):

        labels_ordered = list(self.geom.labels)
        legended = [False, False, False]

        self._PMLs = {}
        if pml_plot:
            figsize = (9.6, 9.72)
            # figsize = (2*_fig.h*_fig.inch2cm*_fig.zoom, 2.5*_fig.h*_fig.inch2cm*_fig.zoom)
            fig, axs = _plt.subplots(figsize=figsize, nrows=3, ncols=1, constrained_layout = True);

        for pml_dom in pml_flags:
            if pml_flags[pml_dom] == True:
                # print(self._map_epsilon[pml_dom])
                for idom in range(len(labels_ordered)):
                    if pml_dom==labels_ordered[idom]:
                        break

                pml_epsilon_raw = self.material._map_epsilon[pml_dom]
                pml_mu_raw = self.material._map_mu[pml_dom]

                # if type(pml_epsilon_raw) is ngs.fem.CoefficientFunction:
                #     pml_epsilon_raw = (pml_epsilon_raw(self.mesh._ngsolve_mesh(self.mesh.cpoint))[0])[0]

                # if type(pml_mu_raw) is ngs.fem.CoefficientFunction:
                #     pml_mu_raw = (pml_mu_raw(self.mesh._ngsolve_mesh(self.mesh.cpoint))[0])[0]

                # print(pml_epsilon_raw)

                pml_x_left = self.geom.intervals[idom]
                pml_x_right = self.geom.intervals[idom+1]
                pml_cpoint = (pml_x_left + pml_x_right)/2
                tPML = pml_x_right - pml_x_left
                nPML = self.geom.nnodes[idom]
                pml_deltax = tPML/nPML

                pml_mPoly = 2
                # pml_R0 = np.exp(-12) # R0: Expected theoretical reflection from PEC backed PML domain
                # pml_sigma0_max = -np.log(pml_R0)*(pml_mPoly+1)/(_const.eta_0*tPML*2)
                pml_sigma0_max = 0.8*(pml_mPoly+1)/(_const.eta_0*pml_deltax*ngs.sqrt(pml_epsilon_raw*pml_mu_raw)) # optimal one

                # alpha_max=0.24 is used in the reference:
                # "S. D. Gedney, Scaled CFS-PML: It Is More Robust, More Accurate,
                # More Efficient, and Simple to Implement. Why Aren’t You Using
                # It?, in 2005 IEEE Antennas and Propagation Society International
                # Symposium, Vol. 4B (IEEE, Washington, DC, USA, 2005), pp. 364–367."
                pml_alpha_max = 0.24 # for fulfilling the causality
                pml_kappa_max = pml_sigma0_max/(self._ngsolve_omega*_const.epsilon_0) # at the current wavelength
                pml_sigma_max = pml_sigma0_max/ngs.sqrt(pml_epsilon_raw*pml_mu_raw)

                # ngs.x is the coordinate variable provided by ngsolve
                # print(pml_cpoint-self.mesh.cpoint[0])
                pml_delta = ngs.IfPos(pml_cpoint-self.mesh.cpoint[0], ngs.x-pml_x_left, pml_x_right-ngs.x)

                ## kappa, sigma, alpha
                # imag(kx)*kappa is integrated to zero, when kappa = 1-2*(pml_delta/tPML)**pml_mPoly
                # pml_kappa_max = -1 # the integral approaches to zero
                pml_kappa_max = 2 # the scaling function approaches to zero
                pml_kappa = 1 + (pml_kappa_max-1)*(pml_delta/tPML)**pml_mPoly
                # pml_kappa = 1 + (pml_kappa_max-1)*(pml_delta/tPML)**(1/pml_mPoly) # inverse polynomial
                # pml_kappa = 0.5 -0.5*(ngs.atan((pml_delta - tPML/2)/(tPML/5))/(np.pi/2)/(np.arctan(5/2)/(np.pi/2)))

                pml_sigma = pml_sigma_max*(pml_delta/tPML)**pml_mPoly
                pml_alpha = pml_alpha_max*((tPML-pml_delta)/tPML)**pml_mPoly

                # Under the convention exp(-j\omega t), it is "+1j" in the denominator
                # in "S. D. Gedney, Introduction to the Finite-Difference Time-Domain (FDTD)
                # Method for Electromagnetics (Springer International Publishing, Cham, 2011)"
                # Here, we stick with the convention exp(i\omega t),
                # so we use "-1j" in the denominator.
                pml_sx = pml_kappa + pml_sigma/(pml_alpha-1j*self._ngsolve_omega*_const.epsilon_0)
                # multiplying pml_alpha by 0 to disable that term for studying leaky modes
                # pml_sx = pml_kappa + 5j*(pml_delta/tPML)**pml_mPoly # a test scaling function

                if pml_plot:
                    pml_xArray = np.linspace(pml_x_left, pml_x_right, nPML)
                    pml_sxArray = np.full_like(pml_xArray, np.nan, dtype=np.complex128)
                    pml_epsilon_rawArray = np.full_like(pml_xArray, np.nan, dtype=np.complex128)
                    pml_mu_rawArray = np.full_like(pml_xArray, np.nan, dtype=np.complex128)

                    # print(pml_mu_rawArray)
                    for ix in np.arange(pml_xArray.size):
                        pml_sxArray[ix] = pml_sx(self.mesh._ngsolve_mesh(pml_xArray[ix]))
                        if type(pml_epsilon_raw) is ngs.fem.CoefficientFunction:
                            pml_epsilon_rawArray[ix] = pml_epsilon_raw(self.mesh._ngsolve_mesh(pml_xArray[ix]))
                        else:
                            pml_epsilon_rawArray[ix] = pml_epsilon_raw
                        if type(pml_mu_raw) is ngs.fem.CoefficientFunction:
                            pml_mu_rawArray[ix] = pml_mu_raw(self.mesh._ngsolve_mesh(pml_xArray[ix]))
                        else:
                            pml_mu_rawArray[ix] = pml_mu_raw

                    # continue
                    axs[0].plot(pml_xArray, np.real(pml_epsilon_rawArray), 'k-', linewidth=0.5, label="$\\epsilon'_{raw}$")
                    axs[0].plot(pml_xArray, np.imag(pml_epsilon_rawArray), 'k--', linewidth=0.5, label="$\\epsilon''_{raw}$")

                    # perpendicular tensor element
                    pml_epsilon_perp = pml_epsilon_rawArray/pml_sxArray
                    axs[0].plot(pml_xArray, np.real(pml_epsilon_perp), 'C0-', linewidth=1, label="$\\epsilon'_{\\perp}$")
                    axs[0].plot(pml_xArray, np.imag(pml_epsilon_perp), 'C0--', linewidth=1, label="$\\epsilon''_{\\perp}$")
                    # parallel tensor element
                    pml_epsilon_prll = pml_epsilon_rawArray*pml_sxArray
                    axs[0].plot(pml_xArray, np.real(pml_epsilon_prll), 'C1-', linewidth=1, label="$\\epsilon'_{\\parallel}$")
                    axs[0].plot(pml_xArray, np.imag(pml_epsilon_prll), 'C1--', linewidth=1, label="$\\epsilon''_{\\parallel}$")

                    if not legended[0]:
                        axs[0].legend(loc="center", fontsize=_fig.fs*1.25)
                        legended[0] = True
                    # axs[0].set_yscale('symlog')

                    # mu
                    axs[2].plot(pml_xArray, np.real(pml_mu_rawArray), 'k-', linewidth=0.5, label="$\\mu'_{raw}$")
                    axs[2].plot(pml_xArray, np.imag(pml_mu_rawArray), 'k--', linewidth=0.5, label="$\\mu''_{raw}$")

                    pml_mu_perp = pml_mu_rawArray/pml_sxArray
                    axs[2].plot(pml_xArray, np.real(pml_mu_perp), 'C0-', label="$\\mu'_{\\perp}$")
                    axs[2].plot(pml_xArray, np.imag(pml_mu_perp), 'C0--', label="$\\mu''_{\\perp}$")
                    pml_mu_prll = pml_mu_rawArray*pml_sxArray
                    axs[2].plot(pml_xArray, np.real(pml_mu_prll), 'C1-', label="$\\mu'_{\\parallel}$")
                    axs[2].plot(pml_xArray, np.imag(pml_mu_prll), 'C1--', label="$\\mu''_{\\parallel}$")

                    if not legended[2]:
                        axs[2].legend(loc="center", fontsize=_fig.fs*1.25)
                        legended[2] = True

                    # n, k is obtained under then harmonic-time convention exp(-i\omega t)
                    # n+ik with a positive k giving attenuation
                    pml_n2_raw = pml_epsilon_rawArray*pml_mu_rawArray
                    pml_n_raw = np.sqrt((np.abs(pml_n2_raw)+pml_n2_raw.real)/2)
                    pml_k_raw = np.sqrt((np.abs(pml_n2_raw)-pml_n2_raw.real)/2)
                    axs[1].plot(pml_xArray, pml_n_raw, 'k-', linewidth=0.5, label="$n_{raw}$")
                    axs[1].plot(pml_xArray, pml_k_raw, 'k--', linewidth=0.5, label="$k_{raw}$")

                    pml_n2_perp = pml_epsilon_perp*pml_mu_perp
                    pml_n_perp = np.sqrt((np.abs(pml_n2_perp)+pml_n2_perp.real)/2)
                    pml_k_perp = np.sqrt((np.abs(pml_n2_perp)-pml_n2_perp.real)/2)
                    axs[1].plot(pml_xArray, pml_n_perp, 'C0-', label="$n_{\\perp}$")
                    axs[1].plot(pml_xArray, pml_k_perp, 'C0--', label="$k_{\\perp}$")
                    pml_n2_prll = pml_epsilon_prll*pml_mu_prll
                    pml_n_prll = np.sqrt((np.abs(pml_n2_prll)+pml_n2_prll.real)/2)
                    pml_k_prll = np.sqrt((np.abs(pml_n2_prll)-pml_n2_prll.real)/2)
                    axs[1].plot(pml_xArray, pml_n_prll, 'C1-', label="$n_{\\parallel}$")
                    axs[1].plot(pml_xArray, pml_k_prll, 'C1--', label="$n_{\\parallel}$")

                    if not legended[1]:
                        axs[1].legend(loc="center", fontsize=_fig.fs*1.25)
                        legended[1] = True

                ## Store PML information
                # in a dictionary with label as the key, and new epsilon, mu and old epsilon, mu
                # self._PMLs |= {pml_dom: [pml_epsilon_xx, pml_epsilon_yy, pml_epsilon_zz, pml_mu_xx, pml_mu_yy, pml_mu_zz, self.material._map_epsilon[pml_dom], self.material._map_mu[pml_dom]]}
                self._PMLs |= {pml_dom: pml_sx}

                ## Update PML domains with new scaling function
                self.material._map_sx[pml_dom] = pml_sx
                # self.material._map_epsilon[pml_dom] = pml_sx
                # self.material._map_mu[pml_dom] = pml_sx

        if pml_plot:
            axs[2].set_xlabel("x (m)")
            axs[0].set_ylabel("$\\epsilon$", fontsize=_fig.fs*1.25)
            axs[1].set_ylabel("$\\tilde{n}$", fontsize=_fig.fs*1.25)
            axs[2].set_ylabel("$\\mu$", fontsize=_fig.fs*1.25)
            axs[0].set_title("PML $\\epsilon$, $\\tilde{n}=(n+ik)$, and $\\mu$")
            if log_plot:
                axs[0].set_yscale("log")
                axs[1].set_yscale("log")
                axs[2].set_yscale("log")

    # setTBC()
    # accepts two numbers defining the indices of the leftmost to the
    # rightmost semi-infinite regions that bounding the core waveguide region.
    # Then, only one element is assigned to each semi-infinite domain, since the
    # field therein has an analytical representation.
    #
    def setTBC(self, tbc_doms=[1,-2]):
        self.TBC = True

        idx_kept = [tbc_doms[0], tbc_doms[1]+1]
        if idx_kept[0]<0:
            idx_kept[0] = None
        if idx_kept[1]==0:
            idx_kept[1] = None

        self.geom_raw = copy.deepcopy(self.geom)
        self.geom.intervals = self.geom.intervals[idx_kept[0]:idx_kept[1]]
        self.geom.nnodes = self.geom.nnodes[idx_kept[0]:idx_kept[1]]
        self.geom.labels = self.geom.labels[idx_kept[0]:idx_kept[1]]

        # only one artificial mesh element is needed for each semi-infinite domain
        self.geom.nnodes[0] = 1
        self.geom.nnodes[-1] = 1

        # update the waveguide geometry
        self.geom.Update(self.geom.intervals, self.geom.nnodes, self.geom.labels)

        # create the mesh
        # define _bnd_left and _bnd_right to the second leftmost and second rightmost boundaries, respectively.
        # The first leftmost and rightmost boundaries arefictitious boundaries
        # for keeping the information of the open semi-infinite domains.
        #
        self.mesh.Create(bnd_left=1, bnd_right=-2)
        # self.mesh.Plot(); sys.exit(0);

        regions = self.mesh._ngsolve_mesh.GetMaterials()
        semi_left = self.mesh._ngsolve_mesh.Materials(regions[0])
        semi_right = self.mesh._ngsolve_mesh.Materials(regions[-1])
        self.mesh._semi = semi_left + semi_right
        self.mesh._wg = ~self.mesh._semi
        # print(self.mesh._wg.Mask()); sys.exit(0);

    def Build(self, fes_order=2, bspl_order=1, ld_scale=1):

        #### Function space creation
        if self.TBC:
            _fes_raw = ngs.H1(self.mesh._ngsolve_mesh, order=fes_order, complex=True, dirichlet=self.mesh._bnd_left+'|'+self.mesh._bnd_right, definedon=self.mesh._wg)
            self.fes = ngs.Compress(_fes_raw)
        else:
            self.fes = ngs.H1(self.mesh._ngsolve_mesh, order=fes_order, complex=True, dirichlet=self.mesh._bnd_left+'|'+self.mesh._bnd_right)

        # trial (for expanding unknowns) and test (for obtaining the weak formulation) functions
        self._u, self._v = self.fes.TnT()

        #### Tabulated materials are represented by ngs.BSpline
        if not self.material._created:
            self.material.Create(bspl_order=bspl_order, ld_scale=ld_scale)
        # self.material._BSplinePlot("agjc"); sys.exit(0);

        # raw material properties
        map_epsilon_re = self.material._map_epsilon.copy()
        map_epsilon_im = self.material._map_epsilon.copy()

        for i, label in enumerate(self.material._map_epsilon):
            if not type(self.material._map_epsilon[label]) is float:
                map_epsilon_re[label] = self.material._map_epsilon[label].real
                map_epsilon_im[label] = self.material._map_epsilon[label].imag
            else:
                map_epsilon_im[label] = 0

        self._cf_epsilon_raw_re = self.mesh._ngsolve_mesh.MaterialCF(map_epsilon_re, default=self.material.default_epsilon)
        self._cf_epsilon_raw_im = self.mesh._ngsolve_mesh.MaterialCF(map_epsilon_im, default=0)
        self._cf_epsilon_raw = self._cf_epsilon_raw_re + 1j*self._cf_epsilon_raw_im

        map_mu_re = self.material._map_mu.copy()
        map_mu_im = self.material._map_mu.copy()
        for i, label in enumerate(self.material._map_mu):
            if not type(self.material._map_mu[label]) is float:
                map_mu_re[label] = self.material._map_mu[label].real
                map_mu_im[label] = self.material._map_mu[label].imag
            else:
                map_mu_im[label] = 0

        self._cf_mu_raw_re = self.mesh._ngsolve_mesh.MaterialCF(map_mu_re, default=self.material.default_mu)
        self._cf_mu_raw_im = self.mesh._ngsolve_mesh.MaterialCF(map_mu_im, default=0)
        self._cf_mu_raw = self._cf_mu_raw_re + 1j*self._cf_mu_raw_im

        #### PML materials setup
        if hasattr(self.material, 'pml'):
            # print(self.material._map_epsilon, self.material._map_mu) # raw materials
            self.SetPML(self.material.pml, pml_plot=False)
            print("Setup PML materials:", self.material.pml)
            # print(self.material._map_epsilon, self.material._map_mu) # update PML domains with pml materials
            # sys.exit(0)
        else:
            print("No PML domain is used!")

        #### define ngsolve CoefficientFunction for the scaling function
        self._cf_sx = self.mesh._ngsolve_mesh.MaterialCF(self.material._map_sx, default=1.0)

        # for deriving any dispersive properties
        self._cf_sx_domega = self._cf_sx.Diff(self._ngsolve_omega)

        # scaled material properties
        # 3 diagonal elements of the epsilon tensor
        self._cf_epsilon = np.array([[self._cf_epsilon_raw/self._cf_sx, 0, 0],
                                     [0, self._cf_epsilon_raw*self._cf_sx, 0],
                                     [0, 0, self._cf_epsilon_raw*self._cf_sx]
                                     ])
        # 3 diagonal elements of the mu tensor
        self._cf_mu = np.array([[self._cf_mu_raw/self._cf_sx, 0, 0],
                                [0, self._cf_mu_raw*self._cf_sx, 0],
                                [0, 0, self._cf_mu_raw*self._cf_sx]
                                ])

        label_left = self.geom.labels[0]
        label_right = self.geom.labels[-1]

        self._epsilon_yy_left = self.material._map_epsilon[label_left]
        self._mu_yy_left = self.material._map_mu[label_left]
        self._epsilon_yy_right = self.material._map_epsilon[label_right]
        self._mu_yy_right = self.material._map_mu[label_right]

        n2_left = self._epsilon_yy_left*self._mu_yy_left
        n2_right = self._epsilon_yy_right*self._mu_yy_right

        self._delta2 = (n2_right-n2_left)
        self._Sigma2 = (n2_right+n2_left)

        self._cf_alpha = [self._cf_epsilon, self._cf_mu]
        self._cf_gamma = [self._cf_mu, self._cf_epsilon]
        self._alpha_yy_right = [self._epsilon_yy_right, self._mu_yy_right]
        self._alpha_yy_left = [self._epsilon_yy_left, self._mu_yy_left]

        #### with TBC
        if self.TBC:
            #### formulated as a polynomial nonlinear eigenvalue problem
            if self.TBC_PEP:
                ## matrices to be used in PEP module
                self._AListTM = [ngs.BilinearForm(self.fes),
                                 ngs.BilinearForm(self.fes,check_unused=False),
                                 ngs.BilinearForm(self.fes),
                                 ngs.BilinearForm(self.fes, check_unused=False),
                                 ngs.BilinearForm(self.fes),
                                 ]

                self._AListTE = [ngs.BilinearForm(self.fes),
                                 ngs.BilinearForm(self.fes,check_unused=False),
                                 ngs.BilinearForm(self.fes),
                                 ngs.BilinearForm(self.fes, check_unused=False),
                                 ngs.BilinearForm(self.fes),
                                 ]
                self._AList = [self._AListTM, self._AListTE]

                # write the weak expression
                for im in range(2):
                    self._AList[im][0] += self._delta2**2*self._ngsolve_k0**2*1/(self._cf_alpha[im][0,0]) \
                        *self._v*self._u*ngs.dx(definedon=self.mesh._wg)

                    self._AList[im][1] += 4j*(
                        self._ngsolve_k0*self._delta2/(self._alpha_yy_right[im]) \
                            *self._v.Trace()*self._u.Trace()*ngs.ds(definedon=self.mesh._bnd_right)
                        -self._ngsolve_k0*self._delta2/(self._alpha_yy_left[im]) \
                            *self._v.Trace()*self._u.Trace()*ngs.ds(definedon=self.mesh._bnd_left))

                    self._AList[im][2] += (
                        -16*(1/(self._cf_alpha[im][1,1])*ngs.grad(self._v)*ngs.grad(self._u)*ngs.dx(definedon=self.mesh._wg) \
                             - self._ngsolve_k0**2*self._cf_gamma[im][2,2]*self._v*self._u*ngs.dx(definedon=self.mesh._wg)) \
                        -8*self._Sigma2*self._ngsolve_k0**2/self._cf_alpha[im][0,0]*self._v*self._u*ngs.dx(definedon=self.mesh._wg))

                    self._AList[im][3] += 16j*(
                        self._ngsolve_k0/(self._alpha_yy_right[im]) \
                            *self._v.Trace()*self._u.Trace()*ngs.ds(definedon=self.mesh._bnd_right)
                        +self._ngsolve_k0/(self._alpha_yy_left[im]) \
                            *self._v.Trace()*self._u.Trace()*ngs.ds(definedon=self.mesh._bnd_left))

                    self._AList[im][4] += 16*self._ngsolve_k0**2/(self._cf_alpha[im][0,0]) \
                        *self._v*self._u*ngs.dx(definedon=self.mesh._wg)

            #### formulated as a split-form nonlinear eigenvalue problem
            else:
                ## matrices to be used in NEP module
                self._AListTM = [ngs.BilinearForm(self.fes),
                                 ngs.BilinearForm(self.fes),
                                 ngs.BilinearForm(self.fes,check_unused=False),
                                 ngs.BilinearForm(self.fes, check_unused=False),
                                 ]

                self._AListTE = [ngs.BilinearForm(self.fes),
                                 ngs.BilinearForm(self.fes),
                                 ngs.BilinearForm(self.fes,check_unused=False),
                                 ngs.BilinearForm(self.fes, check_unused=False),
                                 ]
                self._AList = [self._AListTM, self._AListTE]

                # write the weak expression
                for im in range(2):
                    self._AList[im][0] += 1/self._cf_alpha[im][1,1] \
                        *ngs.grad(self._v)*ngs.grad(self._u)*ngs.dx(definedon=self.mesh._wg) \
                        - self._ngsolve_k0**2*self._cf_gamma[im][2,2] \
                        *self._v*self._u*ngs.dx(definedon=self.mesh._wg)
                    self._AList[im][1] += self._ngsolve_k0**2/self._cf_alpha[im][0,0] \
                        *self._v*self._u*ngs.dx(definedon=self.mesh._wg)
                    self._AList[im][2] += +1j*self._ngsolve_k0/self._alpha_yy_left[im] \
                        *self._v.Trace()*self._u.Trace()*ngs.ds(definedon=self.mesh._bnd_left)
                    self._AList[im][3] += +1j*self._ngsolve_k0/self._alpha_yy_right[im] \
                        *self._v.Trace()*self._u.Trace()*ngs.ds(definedon=self.mesh._bnd_right)

        #### w/o TBC: PML backed by PEC
        else:
            # formulate a linear eigenvalue problem in a weak form
            self._a = [ngs.BilinearForm(self.fes), ngs.BilinearForm(self.fes)]
            self._m = [ngs.BilinearForm(self.fes), ngs.BilinearForm(self.fes)]

            # k0 = omega0/c_const
            for im in range(2):
                self._a[im] += 1/(self._cf_alpha[im][1,1]) \
                    *ngs.grad(self._v)*ngs.grad(self._u)*ngs.dx \
                    -self._ngsolve_k0**2*self._cf_gamma[im][2,2]*self._v*self._u*ngs.dx
                self._m[im] += -1*self._ngsolve_k0**2/self._cf_alpha[im][0,0] \
                    *self._v*self._u*ngs.dx

    class Solution:
        def __init__(self, fes, ndof=1, neigs=1):

            self.neigs = neigs

            self.npoints = len(fes.mesh.ngmesh.Points())
            self.x = [0.0]*self.npoints
            for ip in range(self.npoints):
                self.x[ip], y, z = fes.mesh.ngmesh.Points()[ip+1].p

            # quantities that might be frequency dependent
            self._mu_raw = np.empty((self.npoints), dtype=np.complex128)
            self._mu = np.empty((self.npoints, 3), dtype=np.complex128) # 3 digonal elements in a tensor
            self._epsilon_raw = np.empty((self.npoints), dtype=np.complex128)
            self._epsilon = np.empty((self.npoints, 3), dtype=np.complex128) # 3 digonal elements in a tensor
            self._sx = np.empty((self.npoints), dtype=np.complex128)
            self._n2list = []
            self._nlist = []
            self.k0 = 1.0
            self.w = 1.0

            # results of two modes are aggregated
            # 0: TM mode, 1: TE mode
            self._nvalid = [0, 0] # number of retained eigenvalues
            self._eigval = np.full((2, self.neigs), np.nan+1j*np.nan, dtype=np.complex128)

            # beta: propagation constant
            self.beta = np.full((2, self.neigs), np.nan+1j*np.nan, dtype=np.complex128)

            # neff: beta/k0, effective mode index
            self.neff = np.full((2, self.neigs), np.nan+1j*np.nan, dtype=np.complex128)

            # uz: the field profiles, uz[0,:,:]: TM mode; uz[1,:,:]: TE mode
            self.uz = np.full((2, self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex128)

            # If TBCs are used, save the following quantities as well.
            self.kappa = self.neff

            # kappa2: neff**2
            self.kappa2 = np.full((2, self.neigs), np.nan+1j*np.nan, dtype=np.complex128)

            # transverse wavenumber of the substrate/cover layers
            self.taus = np.full((2, self.neigs), np.nan+1j*np.nan, dtype=np.complex128)
            self.tauc = np.full((2, self.neigs), np.nan+1j*np.nan, dtype=np.complex128)


    def Solve(self, solver="ngsolve_ArnoldiSolver", keepInMem=False, show_pattern=False):

        print("Solving at the normalized frequency w = "+"{0.real:.3f}".format(self.w))

        if self.TBC: # solve with TBC
            # The factor 4 comes from the fact that this is a quartic nonlinear eigenvalue problem
            self.sol = self.Solution(fes=self.fes, ndof=self.fes.ndof, neigs=self.fes.ndof*4)

        else: # solve w/o TBC
            # a normal linear eigenvalue problem
            self.sol = self.Solution(fes=self.fes, ndof=self.fes.ndof, neigs=self.fes.ndof)

        #TODO: Solve eigenvalues less than "ndof*4" or "ndof".
        # self.sol = self.Solution(ndof=self.fes.ndof, neigs=20) # this doesn't work

        ## auxiliary variables, for post-processing and filtering
        # list of complex refractive indices segment by segment
        intervals = np.asarray(self.geom.intervals)
        mid_intervals = 0.5*(intervals[:-1]+intervals[1:])

        for p in mid_intervals:

            mu_raw = self._cf_mu_raw(self.mesh._ngsolve_mesh(p))
            epsilon_raw = self._cf_epsilon_raw(self.mesh._ngsolve_mesh(p))

            if np.abs(mu_raw-1)>1e-4:
                print("Be careful interpreting the refractive index when the permeability is not 1!")
            self.sol._n2list.append(mu_raw*epsilon_raw)
            self.sol._nlist.append(self.sol._n2list[-1]**0.5)

        # For investigating the PML scaling factor sx and epsilon/mu, before and after enabling PML
        for i, p in zip(range(self.mesh.npoints), self.mesh._ngsolve_mesh.ngmesh.Points()):
            self.sol._sx[i] = self._cf_sx(self.mesh._ngsolve_mesh(p[0]))

            self.sol._mu_raw[i] = self._cf_mu_raw(self.mesh._ngsolve_mesh(p[0]))
            self.sol._epsilon_raw[i] = self._cf_epsilon_raw(self.mesh._ngsolve_mesh(p[0]))

            self.sol._mu[i,:] = self.sol._mu_raw[i]*np.array([1/self.sol._sx[i], self.sol._sx[i], self.sol._sx[i]])
            self.sol._epsilon[i,:] = self.sol._epsilon_raw[i]*np.array([1/self.sol._sx[i], self.sol._sx[i], self.sol._sx[i]])

        #%% solve with TBC
        if self.TBC:

            n2left = self.sol._epsilon_raw[0]*self.sol._mu_raw[0]
            n2right = self.sol._epsilon_raw[-1]*self.sol._mu_raw[-1]

            n2_sum = (n2right + n2left)
            n2_diff = (n2right - n2left)
            Sigma2 = n2_sum
            delta2 = n2_diff

            def ngs2petscMatAIJList(ngs_matList):
                petsc_matList = [[], []]
                for im in range(2):
                    for ngs_mat in ngs_matList[im]:
                        #!!! the matrix conversion provided by ngs is wrong,
                        # as the two outermost dimensions are removed.
                        # petsc_matList.append(ngs.ngs2petsc.CreatePETScMatrix(ngs_mat.mat, self.fes.FreeDofs()))

                        petsc_matList[im].append(ngs2petscMatAIJ(ngs_mat.mat))

                return petsc_matList

            def assembleList(AList):
                for im in range(2):
                    for Ai in AList[im]:
                        Ai.Assemble()

            assembleList(self._AList)
            petsc_AList = ngs2petscMatAIJList(self._AList)

            #### investigate the system matrices
            if show_pattern:
                fig, axs = _plt.subplots(figsize = (6*_fig.h*_fig.inch2cm*_fig.zoom, 2.8*_fig.h*_fig.inch2cm*_fig.zoom), ncols = 5, nrows = 4, constrained_layout = True);

                for ic in range(len(petsc_AList[0])):

                    axs[0, ic].set_title("$\mathbf{A}_"+str(ic)+"$")
                    for im in range(2):
                        numpy_A = petscMat2numpyMat(petsc_AList[im][ic])
                        acb = axs[2*im+0, ic].imshow(np.real(numpy_A)); _plt.colorbar(acb, ax=axs[2*im+0,ic]);
                        acb = axs[2*im+1, ic].imshow(np.imag(numpy_A)); _plt.colorbar(acb, ax=axs[2*im+1,ic]);

                for ir in range(2):
                    axs[2*ir+0, 0].set_ylabel("$\mathrm{Re}$")
                    axs[2*ir+1, 0].set_ylabel("$\mathrm{Im}$")

                sys.exit(0)

            # Setup the number of eigensolvers to be sought
            # neigs_query = min(self.fes.ndof, 200) # the number of eigenvalues is potentially self.fes.ndof*4
            neigs_query = self.fes.ndof*4

            # Searching eigenvalues in a contour defined by vertices of a polygon.
            # a and b: largest possible real and imaginary values of the mode index.
            nmax = np.max(np.abs(self.sol._n2list))**0.5
            a = nmax*100.1; b = nmax*100.1;
            vertices = np.array([-a-1j*b, a-1j*b, a+1j*b, -a+1j*b])

            #### SLEPc.PEP()
            if self.TBC_PEP:

                ## a split form
                # self._Q = SLEPc.NEP().create() # try to solve with a general NEP module

                # f0 = SLEPc.FN().create()
                # f0.setType(SLEPc.FN.Type.RATIONAL)
                # f0.setRationalNumerator([1.0])

                # f1 = SLEPc.FN().create()
                # f1.setType(SLEPc.FN.Type.RATIONAL)
                # f1.setRationalNumerator([1.0, 0.0])

                # f2 = SLEPc.FN().create()
                # f2.setType(SLEPc.FN.Type.RATIONAL)
                # f2.setRationalNumerator([1.0, 0.0, 0.0])

                # f3 = SLEPc.FN().create()
                # f3.setType(SLEPc.FN.Type.RATIONAL)
                # f3.setRationalNumerator([1.0, 0.0, 0.0, 0.0])

                # f4 = SLEPc.FN().create()
                # f4.setType(SLEPc.FN.Type.RATIONAL)
                # f4.setRationalNumerator([1.0, 0.0, 0.0, 0.0, 0.0])
                # from petsc4py import PETSc
                # flist = [f0, f1, f2, f3, f4]

                ## a polynomial form
                self._Q = SLEPc.PEP().create()

                # Neglect the frist two lowest order matrices, for the special case of
                # symmetric cover/substrate layers, when \delta^2 is almost equal to zero.
                if np.abs(n2_diff)<1e-8:
                    for im in range(2):
                        del petsc_AList[im][0:2]
                        # if len(flist)>0:
                        #     del flist[3:]

                for im in range(2):

                    ## 1. sovle the split form
                    # self._Q.setSplitOperator(petsc_AList[im][::-1], flist[::-1], PETSc.Mat.Structure.SUBSET)
                    # self._Q.setTolerances(1.0e-8, 50)
                    # self._Q.setDimensions(2) # number of requested eigenvalues
                    # self._Q.setProblemType(SLEPc.NEP.ProblemType.RATIONAL)
                    # # self._Q.setProblemType(SLEPc.NEP.ProblemType.GENERAL)
                    # # self._Q.setType(SLEPc.NEP.Type.RII) # doesn't work so far
                    # # self._Q.setType(SLEPc.NEP.Type.SLP) # doesn't work so far
                    # self._Q.setType(SLEPc.NEP.Type.NARNOLDI) # the most promising solver
                    # # self._Q.setType(SLEPc.NEP.Type.NLEIGS) # doesn't work so far
                    # # self._Q.setType(SLEPc.NEP.Type.CISS)
                    # # self._Q.setType(SLEPc.NEP.Type.INTERPOL) # doesn't work so far

                    # # # eigenvalues in a contour
                    # # # only the elliptic region is implemented in SLEPc.NEP
                    # # rg = self._Q.getRG()
                    # # rg.setType(SLEPc.RG.Type.ELLIPSE)
                    # # rg.setEllipseParameters(a, b, 1)

                    ## 2. sovle the polynomial form
                    self._Q.setOperators(petsc_AList[im])
                    self._Q.setTolerances(1.0e-8, 50)
                    self._Q.setDimensions(neigs_query) # number of requested eigenvalues
                    self._Q.setProblemType(SLEPc.PEP.ProblemType.GENERAL) # other problem type do not work well.

                    slepc_rg = self._Q.getRG()
                    slepc_rg.setType(SLEPc.RG.Type.POLYGON)
                    slepc_rg.setPolygonVertices(vertices)

                    # The default "SLEPc.PEP.Basis.MONOMIAL" basis works the best,
                    # the other bases do not work so far.
                    self._Q.setBasis(SLEPc.PEP.Basis.MONOMIAL)

                    ## solving
                    self._Q.setFromOptions()
                    self._Q.solve()

                    # derived quantities
                    nvalid = min(self._Q.getConverged(), neigs_query)

                    petsc_y = petsc_AList[im][0].createVecs(side='right')
                    eigval = np.full((nvalid), np.nan+1j*np.nan, dtype=np.complex128)
                    kappa2 = np.full((nvalid), np.nan+1j*np.nan, dtype=np.complex128)
                    uz = np.full((nvalid, self.fes.ndof), np.nan+1j*np.nan, dtype=np.complex128)

                    for i in range(nvalid):
                        eigval[i] = self._Q.getEigenpair(i, petsc_y)
                        uz[i, :] = petsc_y.array

                    # save derived results
                    kappa2 = 0.5*Sigma2-eigval**2-delta2**2/(16*eigval**2)

                    ## sort mode by kappa2 values
                    sort_mode = True
                    # sort_mode = False
                    if sort_mode:
                        idx_sort = np.argsort(kappa2)

                    # in a sorted order
                    self.sol._eigval[im, :nvalid] = eigval[idx_sort]
                    self.sol.kappa2[im, :nvalid] = kappa2[idx_sort]
                    self.sol.uz[im, :nvalid, :] = uz[idx_sort, :]

                    # other derived results
                    self.sol.kappa[im, :nvalid] = np.sqrt(self.sol.kappa2[im, :nvalid])
                    # another choice for the Riemann sheet of kappa values
                    # self.sol.kappa[im, :nvalid] = np.abs(self.sol.kappa2[im, :nvalid])**0.5 \
                    #     *np.exp(1j*np.mod(np.angle(self.sol.kappa2[im, :nvalid]), 2*np.pi)/2)

                    self.sol.beta[im, :nvalid] = self.sol.kappa[im, :nvalid]*self.k0
                    #!!! Always double-check the sign conventions of the
                    # transverse wave number in the substrate and cover layers.
                    self.sol.taus[im, :nvalid] = -self.sol._eigval[im, :nvalid] + delta2/(4*self.sol._eigval[im, :nvalid])
                    self.sol.tauc[im, :nvalid] = +self.sol._eigval[im, :nvalid] + delta2/(4*self.sol._eigval[im, :nvalid])
                    self.sol._nvalid[im] = nvalid

                    # print(self.sol.kappa[im, :nvalid])
                    # sys.exit(0)

            #### SLEPc.NEP()
            else:
                self._Q = SLEPc.NEP().create()

                ## split form: functions definition
                f0 = SLEPc.FN().create()
                f0.setType(SLEPc.FN.Type.RATIONAL)
                f0.setRationalNumerator([1.0])

                f1 = SLEPc.FN().create()
                f1.setType(SLEPc.FN.Type.RATIONAL)
                f1.setRationalNumerator([1.0, 0.0, 0.0])

                kx2_left = SLEPc.FN().create()
                kx2_left.setType(SLEPc.FN.Type.RATIONAL)
                kx2_left.setRationalNumerator([-1.0, 0.0, n2left])

                kx2_right = SLEPc.FN().create()
                kx2_right.setType(SLEPc.FN.Type.RATIONAL)
                kx2_right.setRationalNumerator([-1.0, 0.0, n2right])

                f_sqrt = SLEPc.FN().create()
                f_sqrt.setType(SLEPc.FN.Type.SQRT)

                kx_left = SLEPc.FN().create()
                kx_left.setType(SLEPc.FN.Type.COMBINE)
                kx_left.setCombineChildren(SLEPc.FN.CombineType.COMPOSE, kx2_left, f_sqrt)

                kx_right = SLEPc.FN().create()
                kx_right.setType(SLEPc.FN.Type.COMBINE)
                kx_right.setCombineChildren(SLEPc.FN.CombineType.COMPOSE, kx2_right, f_sqrt)

                from petsc4py import PETSc
                for im in range(2):
                    self._Q.setSplitOperator(
                        petsc_AList[im],
                        [f0, f1, kx_left, kx_right],
                        PETSc.Mat.Structure.UNKNOWN)
                    self._Q.setTolerances(1.0e-8)#, 50)
                    self._Q.setDimensions(neigs_query) # number of requested eigenvalues
                    # self._Q.setProblemType(SLEPc.NEP.ProblemType.RATIONAL)
                    self._Q.setProblemType(SLEPc.NEP.ProblemType.GENERAL)
                    # self._Q.setType(SLEPc.NEP.Type.RII) # doesn't work so far
                    # self._Q.setType(SLEPc.NEP.Type.SLP) # gives few results
                    # self._Q.setType(SLEPc.NEP.Type.NARNOLDI) # doesn't work so far
                    # self._Q.setType(SLEPc.NEP.Type.NLEIGS) # doesn't work so far
                    self._Q.setType(SLEPc.NEP.Type.CISS) # this can only find part eigenvalues in a small ellipse, eigenvalues are kind of degenerate
                    # self._Q.setType(SLEPc.NEP.Type.INTERPOL) # doesn't work so far

                    # self._Q.setTarget(1.0)
                    # self._Q.setWhichEigenpairs(SLEPc.NEP.Which.TARGET_MAGNITUDE)

                    # print(self._Q.getCISSSizes())
                    # self._Q.setCISSSizes(ip=64, bs=32, ms=16, npart=1, bsmax=64)
                    # print(self._Q.getCISSSizes())

                    # eps = self._Q.getSLPEPS();
                    # print(eps.getType())
                    # eps.setType(SLEPc.EPS.Type.LAPACK)
                    # eps.setType(SLEPc.EPS.Type.CISS)
                    # print(eps.getType())
                    # print(eps.getProblemType())
                    # eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
                    # print(eps.getProblemType())

                    # rg = eps.getRG()

                    # eigenvalues in a contour
                    # only the elliptic region is implemented in SLEPc.NEP
                    rg = self._Q.getRG()
                    rg.setType(SLEPc.RG.Type.ELLIPSE)
                    # rg.setEllipseParameters(np.max(np.abs(self.sol._nlist)), b/10, 1)
                    rg.setEllipseParameters(2.0, b/100, 1)

                    self._Q.setFromOptions()
                    self._Q.solve()

                    nvalid = self._Q.getConverged()
                    # print("Number of converged eigenpairs %d" % nconv)

                    petsc_y = petsc_AList[im][0].createVecs(side='right')
                    eigval = np.full((nvalid), np.nan+1j*np.nan, dtype=np.complex128)
                    kappa2 = np.full((nvalid), np.nan+1j*np.nan, dtype=np.complex128)
                    uz = np.full((nvalid, self.fes.ndof), np.nan+1j*np.nan, dtype=np.complex128)

                    for i in range(nvalid):
                        eigval[i] = self._Q.getEigenpair(i, petsc_y)
                        uz[i, :] = petsc_y.array

                    print(eigval)
                    sys.exit(0)
                    kappa2 = eigval**2

                    ## sort mode by kappa2 values
                    sort_mode = True
                    # sort_mode = False
                    if sort_mode:
                        idx_sort = np.argsort(kappa2)

                    self.sol._eigval[im, :nvalid] = eigval[idx_sort]
                    self.sol.kappa[im, :nvalid] = eigval[idx_sort]
                    self.sol.kappa[im, :nvalid] = eigval[idx_sort]**2
                    self.sol.uz[im, :nvalid, :] = uz[idx_sort, :]

                    self.sol.beta[im, :nvalid] = self.sol.kappa[im, :nvalid]*self.k0
                    #!!! Always double-check the sign conventions of the
                    # transverse wave number in the substrate and cover layers.
                    self.sol.taus[im, :nvalid] = np.sqrt(n2left-self.sol.kappa2[im, :nvalid])
                    self.sol.tauc[im, :nvalid] = np.sqrt(n2right-self.sol.kappa2[im, :nvalid])
                    self.sol._nvalid[im] = nvalid

        #%% solve w/o TBC
        #
        # spcipy.sparse.linalg provides eigs and eigsh which are wrapped around
        # the ARPACK eigenvalue solvers.
        # https://github.com/scipy/scipy/blob/v1.11.2/scipy/sparse/linalg/_eigen/arpack/arpack.py
        #
        else:
            # for linear eigenvalue problem
            if solver=="ngsolve_ArnoldiSolver":

                # inspect the matrix pattern
                if show_pattern:
                    fig, axs = _plt.subplots(figsize = (2*_fig.h*_fig.inch2cm*_fig.zoom, 1.86*_fig.h*_fig.inch2cm*_fig.zoom), ncols = 2, nrows = 2, constrained_layout = True);

                for im in range(2):
                    self._a[im].Assemble()
                    self._m[im].Assemble()

                    if show_pattern:
                        A = ngs2numpyMat(self._a[im])
                        M = ngs2numpyMat(self._m[im])

                        axs[0,im].imshow(np.abs(A));
                        axs[1,im].imshow(np.abs(M));
                        axs[0,im].set_title("K($A$) ="+"{0.real:.2f}".format(np.linalg.cond(A, p='fro')))
                        axs[1,im].set_title("K($M$) ="+"{0.real:.2f}".format(np.linalg.cond(M, p='fro')))
                        _plt.show();

                        # This reveals that only part of the dofs are linked to nodal points,
                        # which is useful for plotting the field profile
                        for i in range(self.fes.ndof):
                            print (i,":", self.fes.CouplingType(i))

                        sys.exit(0)

                    #### Search eigenvalues around eig_sigma.
                    # pro- and post-processing for this shift is done
                    # internally in the eigenvalue solver
                    self.sol._eig_sigma = 1*self.k0**2 # around the vacuum light line

                    uz = ngs.GridFunction(self.fes, multidim=self.sol.neigs, name='modes_raw')

                    with ngs.TaskManager():
                        self.sol._eigval = ngs.ArnoldiSolver(self._a[im].mat, self._m[im].mat, self.fes.FreeDofs(), list(uz.vecs), shift=self.sol._eig_sigma)

                    #### save results
                    self.sol.neff[im, :] = np.sqrt(self.sol._eigval)
                    self.sol.beta[im, :] = self.sol.neff[im, :]*self.k0

                    for i in range(self.fes.ndof):
                        self.sol.uz[im, i, :] = uz.vec.data[:self.mesh.npoints].FV().NumPy()

        self.sol.k0 = self.k0*1.0
        self.sol.w = self.w*1.0

        return self.sol

    #%% interactively select a mode by clicking at the nearest point in the complex plane
    class modeSelector:
        def __init__(self, axs, sol, axs_names, mode_type=0, nmodes=1, axSpectra=[], axs_profile=[]):

            # axs and axs_names must have equal length
            if len(axs) != len(axs_names):
                print("axs and axs_names have different lengths!")
                sys.exit(0)

            nvalid = sol._nvalid[mode_type]
            self.pts = np.full((len(axs), nvalid), np.nan+1j*np.nan, dtype=np.complex128)
            for ia in range(len(axs_names)):
                match axs_names[ia]:

                    case "beta":
                        self.pts[ia, :] = sol.kappa[mode_type,:nvalid]*sol.k0
                    case "kappa" | "neff":
                        self.pts[ia, :] = sol.kappa[mode_type,:nvalid]
                    case "kappa2":
                        self.pts[ia, :] = sol.kappa2[mode_type,:nvalid]
                    case "taus":
                        self.pts[ia, :] = sol.taus[mode_type,:nvalid]
                    case "tauc":
                        self.pts[ia, :] = sol.tauc[mode_type,:nvalid]
                    case "eig" | "eigval":
                        self.pts[ia, :] = sol._eigval[mode_type,:nvalid]
                    case _:
                        print("Only names beta | kappa | neff | kappa2 | taus | tauc | eig | eigval are allowed!")
                        sys.exit(0)

            self.axs = axs
            self.sol = sol
            self.mode_type = mode_type
            self.nmodes = nmodes
            self.axSpectra = axSpectra
            self.axs_profile = axs_profile

            self.axs_pt = []
            self.axs_band = []
            self.axs_uz = []
            self.imode = 0

            self.c_hz = ["C0", "C1", "C3", "C9"]

        def mpl_connect(self):
            self.cid = []
            for ia, ax in zip(range(len(self.axs)), self.axs):
                # initialize each event caller and
                # connect the event callback function to the event manager
                self.cid.append(ax.figure.canvas.mpl_connect('button_press_event', self.onclick))

        def onclick(self, event):

            for ia, ax in zip(range(len(self.axs)), self.axs):
                if event.inaxes==ax.axes:
                    # get the clicking point
                    self.pt = event.xdata + 1j*event.ydata

                    xlim = ax.axes.get_xlim()
                    ylim = ax.axes.get_ylim()
                    x_delta = 0.015*(xlim[1]-xlim[0])
                    y_delta = 0.015*(ylim[1]-ylim[0])

                    # get the mode(s) whose value(s) is/are closest
                    # (within an ellipse) to the clicking point
                    idx = np.where((
                        np.abs(self.pts[ia,:].real - self.pt.real)**2/x_delta**2
                        +np.abs(self.pts[ia,:].imag - self.pt.imag)**2/y_delta**2
                        )<1)[0]

                else:
                    continue

            if len(idx)>0:
                self.plotModes(idx)
            else:
                print("Click closer to a mode!")

        def plotModes(self, idx):

            if len(np.shape(idx))==0:
                self.idx = np.asarray([idx])
            elif len(np.shape(idx))==1:
                self.idx = np.asarray(idx)

            if self.imode>=self.nmodes:

                for ip in range(len(self.axs_pt)):
                    self.axs_pt[ip].remove()
                self.axs_pt = []

                for ib in range(len(self.axs_band)):
                    self.axs_band[ib].remove()
                self.axs_band = []

                for iu in range(len(self.axs_uz)):
                    self.axs_uz[iu].remove()
                self.axs_uz = []

                self.imode = 0

            self.imode += 1

            # axs for plotting field profiles
            if np.size(self.axs_profile)==0:
                fig_tmp, axs_profile = _plt.subplots(figsize=(10, 3), nrows=self.nmodes, ncols=1, layout='constrained')
            else:
                axs_profile = self.axs_profile

            k0 = self.sol.k0
            w = self.sol.w
            nlist = np.asarray(self.sol._nlist)
            n2list = np.asarray(self.sol._n2list)
            n2s, n2c = n2list[[0, -1]]
            # nmax = np.max(nlist.real)
            x = np.asarray(self.sol.x)

            kappa = self.sol.kappa[self.mode_type, :self.sol._nvalid[self.mode_type]]
            taus = self.sol.taus[self.mode_type, :self.sol._nvalid[self.mode_type]]
            tauc = self.sol.tauc[self.mode_type, :self.sol._nvalid[self.mode_type]]

            # mark the selected point(s) in the complex plane(s)
            for ia, ax in zip(range(len(self.axs)), self.axs):
                ax_pt,  = ax.axes.plot(self.pts[ia, self.idx].real, self.pts[ia, self.idx].imag, self.c_hz[self.imode-1]+"*", fillstyle="full", ms=_fig.ms*2)
                self.axs_pt.append(ax_pt)
                ax.figure.canvas.draw()

            # mark the selected point(s) on the dispersion curve
            if np.size(self.axSpectra)==1:
                # mark selected modes in the dispersion curve
                ax_band, = self.axSpectra.plot(np.abs(kappa[self.idx].real)*k0, np.repeat(w, self.idx.size), self.c_hz[self.imode-1]+"*", fillstyle="full", ms=_fig.ms*2)
                self.axs_band.append(ax_band)

                self.axSpectra.figure.canvas.draw()

            ax_profile = axs_profile[self.imode-1]
            for ia, id in zip(range(self.idx.size), self.idx):

                # zf_profile = nmax
                zf_profile = 1

                # get the field profile
                hz = self.sol.uz[self.mode_type, id, 0:self.sol.npoints-2]
                self.hz = hz
                imax_hz = np.nanargmax(np.abs(hz))
                hz *= 1/hz[imax_hz] * zf_profile

                epsilon = self.sol._epsilon_raw[1:-1]
                epsilon[0] = epsilon[1]
                x_s = np.linspace(x[0], x[1], 51, endpoint=True)
                x_c = np.linspace(x[-2], x[-1], 51, endpoint=True)
                hz_s = hz[0]*np.exp(1j*taus[id]*k0*(x_s-x_s[-1]))
                hz_c = hz[-1]*np.exp(1j*tauc[id]*k0*(x_c-x_c[0]))

                Sy = 0.5*(
                    kappa[id].real*epsilon.real
                    +kappa[id].imag*epsilon.imag
                    )*np.abs(hz)**2/np.abs(epsilon)**2

                Sy_s = 0.5*np.exp(-2*taus[id].imag*k0*(x_s-x_s[-1]))*(
                    kappa[id].real*n2s.real
                    +kappa[id].imag*n2s.imag
                    )*np.abs(hz[0])**2/np.abs(n2s)**2

                Sy_c = 0.5*np.exp(-2*tauc[id].imag*k0*(x_c-x_c[0]))*(
                    kappa[id].real*n2c.real
                    +kappa[id].imag*n2c.imag
                    )*np.abs(hz[-1])**2/np.abs(n2c)**2

                Sx_s = 0.5*np.exp(-2*taus[id].imag*k0*(x_s-x_s[-1]))*(
                    taus[id].real*n2s.real
                    +taus[id].imag*n2s.imag
                    )*np.abs(hz[0])**2/np.abs(n2s)**2

                Sx_c = 0.5*np.exp(-2*tauc[id].imag*k0*(x_c-x_c[0]))*(
                    tauc[id].real*n2c.real
                    +tauc[id].imag*n2c.imag
                    )*np.abs(hz[-1])**2/np.abs(n2c)**2

                normalize_power = True
                # normalize_power = False
                if normalize_power:
                    imax_Sy = np.nanargmax(np.abs(Sy))
                    imax_Sy_s = np.nanargmax(np.abs(Sy_s))
                    imax_Sy_c = np.nanargmax(np.abs(Sy_c))
                    imax_Sx_s = np.nanargmax(np.abs(Sx_s))
                    imax_Sx_c = np.nanargmax(np.abs(Sx_c))
                    Sy *= 1/np.abs(Sy[imax_Sy]) * zf_profile
                    Sy_s *= 1/np.abs(Sy_s[imax_Sy_s]) * zf_profile
                    Sy_c *= 1/np.abs(Sy_c[imax_Sy_c]) * zf_profile
                    Sx_s *= 1/np.abs(Sx_s[imax_Sx_s]) * zf_profile
                    Sx_c *= 1/np.abs(Sx_c[imax_Sx_c]) * zf_profile

                # hz field
                ax_uz, = ax_profile.axes.plot(x[1:-1]*1e3, np.real(hz), self.c_hz[self.imode-1]+"-", lw=1.0, ms=_fig.ms); self.axs_uz.append(ax_uz)
                ax_uz, = ax_profile.axes.plot(x_s*1e3, np.real(hz_s), self.c_hz[self.imode-1]+"-", lw=1.0, ms=_fig.ms); self.axs_uz.append(ax_uz)
                ax_uz, = ax_profile.axes.plot(x_c*1e3, np.real(hz_c), self.c_hz[self.imode-1]+"-", lw=1.0, ms=_fig.ms); self.axs_uz.append(ax_uz)
                ax_uz, = ax_profile.axes.plot(x[1:-1]*1e3, np.imag(hz), self.c_hz[self.imode-1]+":", lw=1.0, ms=_fig.ms); self.axs_uz.append(ax_uz)
                ax_uz, = ax_profile.axes.plot(x_s*1e3, np.imag(hz_s), self.c_hz[self.imode-1]+":", lw=1.0, ms=_fig.ms); self.axs_uz.append(ax_uz)
                ax_uz, = ax_profile.axes.plot(x_c*1e3, np.imag(hz_c), self.c_hz[self.imode-1]+":", lw=1.0, ms=_fig.ms); self.axs_uz.append(ax_uz)

                # Poynting vector components
                ax_uz, = ax_profile.axes.plot(x[1:-1]*1e3, np.real(Sy), "C2-", lw=1.5, ms=_fig.ms, clip_on=True); self.axs_uz.append(ax_uz)
                ax_uz, = ax_profile.axes.plot(x_s*1e3, Sy_s, "C2-", lw = 1.5, ms=_fig.ms, clip_on=True); self.axs_uz.append(ax_uz)
                ax_uz, = ax_profile.axes.plot(x_c*1e3, Sy_c, "C2-", lw = 1.5, ms=_fig.ms, clip_on=True); self.axs_uz.append(ax_uz)
                ax_uz, = ax_profile.axes.plot(x_s*1e3, Sx_s, "C4:", lw = 1.5, ms=_fig.ms, clip_on=True); self.axs_uz.append(ax_uz)
                ax_uz, = ax_profile.axes.plot(x_c*1e3, Sx_c, "C4:", lw = 1.5, ms=_fig.ms, clip_on=True); self.axs_uz.append(ax_uz)

                ## print more information
                #-- refractive index
                # ax_profile.step(model.geom.intervals[:-1], nlist.real, where='post', label="n'")
                # ax_profile.step(model.geom.intervals[:-1], nlist.imag, where='post', label="n''")

                #-- solutions
                # ax_profile.set_ylim([-np.max(nlist.real)-0.1, np.max(nlist.real)+0.1])
                # ax_profile.set_title(
                #     "$\\kappa$="+f"{kappa[id]:.4E}"+",\t"
                #     +"$\\lambda$="+f"{kappa[id]:.4E}"+",\t"
                #     +"$\\tau_s$="+f"{taus[id]:.4E}"+",\t"
                #     +"$\\tau_c$="+f"{tauc[id]:.4E}")

                ax_profile.set_yticks([-1, 1])
                ax_profile.set_ylim([-1.05, 1.05])

                ax_profile.figure.canvas.draw()

    #%% access control for some key properties
    # Access control for some variables
    class _ld0_target_access:
        def __get__(self, obj, objtype=None):
            obj._ld0_target = obj._ngsolve_ld0_target.Get()
            return obj._ld0_target
        def __set__(self, obj, val):
            obj._ld0_target = val
            obj._ngsolve_ld0_target.Set(obj._ld0_target)

    class _w_access:
        def __get__(self, obj, objtype=None):
            obj._w = obj._ngsolve_w.Get()
            return obj._w
        def __set__(self, obj, val):
            obj._w = val
            obj._ngsolve_w.Set(obj._w)

    # wavelength access control: ld0 = ld0_target/w
    class _ld0_access:
        def __get__(self, obj, objtype=None):
            obj._ld0 = obj.ld0_target/obj.w
            return obj._ld0
        def __set__(self, obj, val):
            obj._ld0 = val
            obj.w = obj.ld0_target/obj._ld0

    # wave number access control: 2*np.pi/(ld0_target/w)
    class _k0_access:
        def __get__(self, obj, objtype=None):
            obj._k0 = 2*np.pi/(obj.ld0_target/obj.w)
            return obj._k0
        def __set__(self, obj, val):
            obj._k0 = val
            obj.w = obj.ld0_target/(2*np.pi/obj._k0)

    class _ld0Array_access:
        def __get__(self, obj, objtype=None):
            obj._ld0Array = obj.ld0_target/obj.wArray
            return obj._ld0Array
        def __set__(self, obj, val):
            obj._ld0Array = val
            obj.wArray = obj.ld0_target/obj._ld0Array

    #---- Some ngsolve Parameters or CoefficientFunctions, that will be used for ngs.
    # Setting _ngsolve_w as a ngsolve Parameter is the way to interact with ngsolve,
    # though it may be overwritten by user.
    # Here, _ngsolve_w got updated while either of w, ld0, and k0 is changed.
    #
    _ngsolve_w = ngs.Parameter(1.0)
    #
    # # @property # read-only, but not work with ngsolve, cannot be overwritten into other type or value
    # def _ngsolve_w(self): # prefix "_" indicates a private variable
        # return ngs.Parameter(self.w) # Later get updated by _ngsolve_w.Set(w)

    _ngsolve_ld0_target = ngs.Parameter(780e-9)

    # Interdependent variables, that can be modified via either one of them.
    # Spectral information, everything is anchored to the normalized frequency.
    # w: normalized frequency ld0_target/ld0 and k0 = 2*np.pi/ld0, with d_target==ld0_target
    #
    w = _w_access()
    ld0 = _ld0_access()
    k0 = _k0_access()

    # a range of normalized frequencies
    wArray = np.array([1.0])
    ld0Array = _ld0Array_access()

    # The target wavelength, independent variable
    ld0_target = _ld0_target_access()

    # Derived ngsolve CoefficientFunctions
    # for being used by other CoefficientFunctions.
    _ngsolve_omega = 2*np.pi*_const.c*_ngsolve_w/_ngsolve_ld0_target
    _ngsolve_k0 = 2*np.pi/(_ngsolve_ld0_target/_ngsolve_w)

    def __init__(self, w=1.0, intervals=(0, 1), nnodes=(17, 0), labels=("freespace", "dummy")):

        self._ngsolve_w.Set(w)

        self.w = self._ngsolve_w.Get()

        self.geom = self.Geometry(intervals, nnodes, labels)

        # The mesh creation needs the geometry node, so a pointer to geom is given as an argument.
        self.mesh = self.Mesh(self.geom)

        self.material = self.Material(self)

        self.TBC = False
        self.TBC_PEP = True
        # self.TBC_PEP = False
