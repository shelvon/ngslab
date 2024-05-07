#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:15:53 2023
Last modification on Fri Feb 2 10:00:00 2024

@author: shelvon
@email: xiaorun.zang@outlook.com

"""

# physical constants
from scipy.constants import c as const_c
from scipy.constants import mu_0 as const_mu_0
from scipy.constants import epsilon_0 as const_epsilon_0
const_eta_0 = (const_mu_0/const_epsilon_0)**0.5

import ngsolve as ngs

import sys
import copy

import slepc4py
slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc

import numpy as np
import scipy.sparse as scipy_sparse

import matplotlib.pyplot as _plt

class __FigureProperty__:
    inch2cm = 1.0/2.54  # inches-->centimeters
    zoom = 1.0
    h = 8.4 # width of a figure occupying one column in a two-column paper
    fs = 8;
    _plt.rc('font', size=fs)

_fig = __FigureProperty__()

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

            mid_points = 0.5*(np.asarray(self._obj_geom.intervals[:-1]) + np.asarray(self._obj_geom.intervals[1:]))
            for ip, mp in zip(range(mid_points.size), mid_points):
                self._ax.text(mp, 0.05, self._obj_geom.labels[ip], ha="center")

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
                epsilon=n**2
                wl=self._tabulated[il][1]*1e6
                # complex n
                # _plt.plot(n.real, wl, '-', lw=1, label='n');
                # _plt.plot(n.imag, wl, '--', lw=1, label='k');
                # complex epsilon
                # _plt.plot(epsilon.real, wl, ':', lw=1, label="$\\epsilon'$");
                # _plt.plot(epsilon.imag, wl, '-.', lw=1, label="$\\epsilon''$");

                _plt.plot(n.real, n.imag, '-o', lw=1, label='n');
                # _plt.plot(epsilon.real, epsilon.imag, '.', lw=1, label="$\\epsilon'$");
                # self._ax.axis("equal")

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

                    if type(val)==str:

                        # Is it necessary to have recursive string chain?
                        # val = self._resolve("nk", self.nk[label]) # search recursively?

                        for il in range(len(self._ngsolve_BSplines)):
                            label_created = self._ngsolve_BSplines[il][0]
                            if label_created==val:
                                val = self._ngsolve_BSplines[il][2]
                                resolved = True
                                break
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

                    if type(val)==str:

                        for il in range(len(self._ngsolve_BSplines)):
                            label_created = self._ngsolve_BSplines[il][0]
                            if label_created==val:
                                val = self._ngsolve_BSplines[il][2]
                                resolved = True
                                break
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
                    if type(self.nk[label])==str:
                        self.refLoad(self.nk[label])

                # dispersive mu (relative permeability)
                for label in self.mu:
                    if type(self.mu[label])==str:
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
                nk_plot = np.full_like(wArray_plot, np.nan, dtype=np.complex64)
                epsilon_plot = np.full_like(wArray_plot, np.nan, dtype=np.complex64)

                for iw in range(wArray_plot.size):
                    self._model._ngsolve_w.Set(wArray_plot[iw])
                    mesh_pt = self._model.mesh._ngsolve_mesh(self._model.mesh.cpoint)
                    nk_plot[iw] = (nk_bspl(mesh_pt)[0])[0]
                    epsilon_plot[iw] = (epsilon_bspl(mesh_pt)[0])[0]

                #### Fixme: it is not clear why the last value is not represented correctly
                # ax.plot(nk_plot.real, wArray_plot, '-o')
                # ax.plot(nk_plot.imag, wArray_plot, '--<')
                # This temporarily hides the last values
                # ax.plot(nk_plot[:-2].real, wArray_plot[:-2], 'C0-', label="n")
                # ax.plot(nk_plot[:-2].imag, wArray_plot[:-2], 'C0-.', label="k")
                # ax.plot(epsilon_plot[:-2].real, wArray_plot[:-2], 'C1--', label="$\\epsilon'$")
                # ax.plot(epsilon_plot[:-2].imag, wArray_plot[:-2], 'C1:', label="$\\epsilon''$")
                # ax.set_xlabel("n (-), k (--), $\\epsilon$ [real (-.), imag (:)]")
                # ax.set_ylabel("w ($\\lambda_{\\mathrm {tg}}/\\lambda$)", fontsize=_fig.fs*1.25)
                ax.legend();


                kArray_plot = 2*np.pi/self._model.ld0_target*wArray_plot
                ax.plot(nk_plot[:-2].real*kArray_plot[:-2], wArray_plot[:-2], 'C0-', label="n")
                ax.plot(nk_plot[:-2].imag*kArray_plot[:-2], wArray_plot[:-2], 'C0-.', label="k")

                ax.plot(1*kArray_plot[:-2], wArray_plot[:-2], 'k-', label="light line")
                ax.plot(1.5*kArray_plot[:-2], wArray_plot[:-2], 'k-', label="light line")
                # ax.plot(1**2*wArray_plot[:-2]**0, wArray_plot[:-2], 'k-', label="light line")
                # ax.plot(1.5**2*wArray_plot[:-2]**0, wArray_plot[:-2], 'k-', label="light line")
                # ax.plot(0*wArray_plot[:-2]**0, wArray_plot[:-2], 'k--', label="zero")
                ax.set_ylim([0, wArray_plot[-2]])
                # ax.set_xlim([-10, 10])

            self._model._ngsolve_w.Set(_ngsolve_w_raw)

    class Solution:
        def __init__(self, neigs=1, ndof=1):

            self.neigs = neigs
            self.dispersive = False

            # TM modes
            self.beta_TM = np.full((self.neigs), np.nan+1j*np.nan, dtype=np.complex64)
            self.neff_TM = np.full((self.neigs), np.nan+1j*np.nan, dtype=np.complex64)
            self.epsilon_TM = np.full((self.neigs), np.nan+1j*np.nan, dtype=np.complex64)

            self.hz_TM = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)
            self.ex_TM = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)
            self.ey_TM = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)

            self.Sy_TM = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)
            self.Sx_TM = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)
            self.ue_TM = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)
            self.um_TM = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)
            self.Qe_TM = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)
            self.Qm_TM = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)

            self.kappa_TM = np.zeros((self.neigs), dtype=float)
            self.kappa_left_TM = np.zeros((self.neigs), dtype=float)
            self.kappa_right_TM = np.zeros((self.neigs), dtype=float)
            self.kappa_wdiff_TM = np.zeros((self.neigs), dtype=float)
            self.kappa_left_wdiff_TM = np.zeros((self.neigs), dtype=float)
            self.kappa_right_wdiff_TM = np.zeros((self.neigs), dtype=float)
            self.filtered_TM = np.full((self.neigs), False)
            self.kappa_Syleft_TM = np.zeros((self.neigs), dtype=float)
            self.kappa_Syright_TM = np.zeros((self.neigs), dtype=float)

            # TE modes
            self.beta_TE = np.full((self.neigs), np.nan+1j*np.nan, dtype=np.complex64)
            self.neff_TE = np.full((self.neigs), np.nan+1j*np.nan, dtype=np.complex64)
            self.epsilon_TE = np.full((self.neigs), np.nan+1j*np.nan, dtype=np.complex64)

            self.ez_TE = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)
            self.hx_TE = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)
            self.hy_TE = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)

            self.Sy_TE = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)
            self.Sx_TE = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)
            self.ue_TE = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)
            self.um_TE = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)
            self.Qe_TE = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)
            self.Qm_TE = np.full((self.neigs, ndof), np.nan+1j*np.nan, dtype=np.complex64)

            self.kappa_TE = np.zeros((self.neigs), dtype=float)
            self.kappa_left_TE = np.zeros((self.neigs), dtype=float)
            self.kappa_right_TE = np.zeros((self.neigs), dtype=float)
            self.kappa_wdiff_TE = np.zeros((self.neigs), dtype=float)
            self.kappa_left_wdiff_TE = np.zeros((self.neigs), dtype=float)
            self.kappa_right_wdiff_TE = np.zeros((self.neigs), dtype=float)
            self.filtered_TE = np.full((self.neigs), False)
            self.kappa_Syleft_TE = np.zeros((self.neigs), dtype=float)
            self.kappa_Syright_TE = np.zeros((self.neigs), dtype=float)

            self.kappa_threshold = 0.5 # ratio of the energy stored in the physical/entire regions

            self.idx_TM = []
            self.idx_TE = []

    class Figure:
        def __init__(self, obj_model):
            self._model = obj_model # a pointer to the model instance
            self.showlegend = False

        def __call__(self, nmodes_plot=4, digits=4, linking=False, debugging=False):
            self.nmodes_plot = nmodes_plot
            self.digits = digits
            self.linking = linking
            self.debugging = debugging

            self.neff_TM =  np.full([self._model.wArray.size, self.nmodes_plot], np.nan+1j*np.nan, dtype=np.complex64)
            self.neff_TE =  np.full([self._model.wArray.size, self.nmodes_plot], np.nan+1j*np.nan, dtype=np.complex64)

            # self._Create_plotly() # initialize the plotly figure
            self.type = "plotly"

            self._Create_matplotlib(self.nmodes_plot) # initialize the matplotlib figure
            self.type = "matplotilb"

        def _frame_args(self, duration):
            return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }
        def _Create_matplotlib(self, nmodes_plot):
            neff_bound = [0.95, 2.05]
            beta_bound = [neff_bound[0]*2*np.pi/self._model.ld0Array[0],
                          neff_bound[1]*2*np.pi/self._model.ld0Array[-1]]
            w_bound = [self._model.wArray[0]-0.1, self._model.wArray[-1]+0.1]

            # figsize = (9.6, 9.72) # fully height occupying a half of the screen
            figsize = (3*_fig.h*_fig.inch2cm*_fig.zoom,
                       2.3*_fig.h*_fig.inch2cm*_fig.zoom)
            self.fig = _plt.figure(figsize = figsize, constrained_layout = True);
            # _plt.ion() # update the figure even in command line

            self.axs = [_plt.subplot2grid((nmodes_plot, 3), (0, 0), rowspan=nmodes_plot, colspan=1)]
            for id in range(nmodes_plot):
                self.axs.extend([_plt.subplot2grid((nmodes_plot, 3), (id, 1), rowspan=1, colspan=1),
                                _plt.subplot2grid((nmodes_plot, 3), (id, 2), rowspan=1, colspan=1)])
                # self.axs[1+id*2].xaxis.set_ticks_position('top')
                # self.axs[2+id*2].xaxis.set_ticks_position('top')

            # plot the light line in vacuum
            self.axs[0].set_xlim([0, 4.0e7])
            self.axs[0].grid(True)
            self.axs[0].set_xlabel("$\\beta_{\mathrm{eff}}$")
            self.axs[0].set_ylabel('$\\frac{\omega}{\omega_{\mathrm{tg}}}$', rotation='horizontal', fontsize=_fig.fs*1.25)#, labelpad=0.5*xdelta)
            self.axs[0].yaxis.set_label_coords(-0.15, 0.65)
            # self.axs[0].set_ylim([self._model.wArray[0]-wDelta, self._model.wArray[-1]+wDelta])
            self.axs[0].plot(2*np.pi/self._model.ld0Array, self._model.wArray, "-", linewidth=1.5, color="gray")


        def _Create_plotly(self):
            neff_bound = [0.95, 2.05]
            beta_bound = [neff_bound[0]*2*np.pi/self._model.ld0Array[0],
                          neff_bound[1]*2*np.pi/self._model.ld0Array[-1]]
            w_bound = [self._model.wArray[0]-0.1, self._model.wArray[-1]+0.1]

            specs_mode_cols = [[{}, {}]]
            pixels_row = [300] + [100]*self.nmodes_plot + [25] + [75]
            pixels_fig = sum(pixels_row)

            row_heights = [pixels_row[0]/pixels_fig]
            for i in range(self.nmodes_plot):
                specs_mode_cols += [[{"secondary_y": True},
                                     {"secondary_y": True}]]
                # subplot_titles += [r"$x$", r"$x$"]
                row_heights += [pixels_row[1]/pixels_fig]

            # figure placeholder for plotting the results
            self.plotly_fig = plotly_subplots.make_subplots(
                rows=1+self.nmodes_plot, cols=2,
                # shared_yaxes = "rows", # Share yaxes in the same row
                row_heights = row_heights,
                specs = specs_mode_cols,
                # subplot_titles=subplot_titles,
                )

            # Set theme, margin, and annotation in layout
            self.plotly_fig.update_layout(
                template="simple_white",
                margin=dict(t=25, b=75, l=10, r=10),
                width=550,
                height=pixels_fig,
            )

            # generate traces
            self.traces_TM = [plotly_go.Scatter(
                mode='markers',
                x=self.neff_TM[:, i].real*2*np.pi/self._model.ld0Array,
                y=self._model.wArray,
                xaxis="x",
                yaxis="y",
                name="TM"+str(i)+" ("+f"{0+0j:+.{self.digits}E})",
                # hovermode="x unified",
                hovertemplate = '%{x:.4g}<extra></extra>',
                marker_color=D3[np.mod(i, len(D3))] if self.linking else D3[0],
                marker_symbol="circle",
                showlegend=self.showlegend,
                ) for i in range(self.nmodes_plot)]
            self.traces_TE = [plotly_go.Scatter(
                mode='markers',
                x=self.neff_TE[:, i].real*2*np.pi/self._model.ld0Array,
                y=self._model.wArray,
                xaxis="x",
                yaxis="y",
                name="TE"+str(i)+" ("+f"{0+0j:+.{self.digits}E})",
                hovertemplate = '%{x:.4g}<extra></extra>',
                marker_color=D3[np.mod(i, len(D3))] if self.linking else D3[0],
                marker_symbol="square",
                showlegend=self.showlegend,
                ) for i in range(self.nmodes_plot)]

            self.traces_hz = [plotly_go.Scatter(
                mode='lines',
                x=self._model.mesh.px, y=self._model.mesh.px*0,
                name="TM"+str(i),
                # text="neffTM"+str(i),
                # textposition="bottom left",
                hoverinfo='skip',
                line=dict(color=D3[np.mod(i, len(D3))] if self.linking else D3[0], width=1, dash='solid'),
                marker_color=D3[np.mod(i, len(D3))] if self.linking else D3[0],
                marker_symbol="circle",
                showlegend=False,
                ) for i in range(self.nmodes_plot)]
            self.traces_ez = [plotly_go.Scatter(
                mode='lines',
                x=self._model.mesh.px, y=self._model.mesh.px*0,
                name="TE"+str(i),
                # text="neffTE"+str(i),
                # textposition="bottom left",
                hoverinfo='skip',
                line=dict(color=D3[np.mod(i, len(D3))] if self.linking else D3[0], width=1.5, dash='dot'),
                marker_color=D3[np.mod(i, len(D3))] if self.linking else D3[0],
                marker_symbol="square",
                showlegend=False,
                ) for i in range(self.nmodes_plot)]

            # plot the ith modal dispersion curve
            # self.plotly_fig.data[i], i in range(nmodes_plot)
            self.itrace_TM = 0
            self.itraces_TM = list(range(self.itrace_TM, self.itrace_TM+self.nmodes_plot))
            for i in range(self.nmodes_plot):
                self.plotly_fig.add_trace(self.traces_TM[i], row=1, col=1)

            self.itrace_TE = self.itrace_TM + self.nmodes_plot
            self.itraces_TE = list(range(self.itrace_TE, self.itrace_TE+self.nmodes_plot))
            for i in range(self.nmodes_plot):
                self.plotly_fig.add_trace(self.traces_TE[i], row=1, col=1)

            # plot the light line
            self.trace_light_line = plotly_go.Scatter(
                mode='lines',
                showlegend=False,
                # mode='lines+markers',
                x=2*np.pi/self._model.ld0Array, y=self._model.wArray,
                marker=dict(size=2,line=dict(width=2,color='black')),
                line_color="black", name="light line",
                )
            self.itrace_light_line = self.itrace_TE + self.nmodes_plot
            self.plotly_fig.add_trace(self.trace_light_line, row=1, col=1)

            self.plotly_fig.update_xaxes(
                title_text=r'$\beta$',
                title_standoff = 0,
                side="bottom",
                range=beta_bound,
                showspikes=True,
                row=1, col=1)
            self.plotly_fig.update_yaxes(
                title_text=r'$\omega/\omega_{\mathrm{tg}}$',
                title_standoff = 0,
                minor_ticks="outside",
                range=w_bound,
                showspikes=True,
                domain=[(sum(pixels_row[1:self.nmodes_plot])+340)/pixels_fig, 1],
                row=1, col=1)

            # plot the silver's epsilon values
            self._model.material.refLoad("agjc")
            x = self._model.material._tabulated[0][2][::-1]
            y = self._model.ld0_target/self._model.material._tabulated[0][1][::-1]

            self.trace_epsilon_real = plotly_go.Scatter(
                x=x.real, y=y,
                showlegend=False,
                line=dict(color='gray', width=1, dash='dot'),
                name=r"$\epsilon '$",
                hovertemplate = '%{x:.4f}<extra></extra>',
                xaxis="x2")
            self.trace_epsilon_imag = plotly_go.Scatter(
                x=x.imag, y=y,
                showlegend=False,
                line=dict(color='gray', width=1, dash='dashdot'),
                name=r"$\epsilon ''$",
                hovertemplate = '%{x:.4f}<extra></extra>',
                xaxis="x2")

            self.itrace_epsilon_real = self.itrace_light_line + 1
            self.plotly_fig.add_trace(self.trace_epsilon_real)#, row=1, col=2)

            self.itrace_epsilon_imag = self.itrace_epsilon_real + 1
            self.plotly_fig.add_trace(self.trace_epsilon_imag)

            self.plotly_fig.update_layout(
                xaxis2=dict(
                    title=r"$\epsilon', \epsilon''$",
                    title_standoff = 0,
                    anchor="y",
                    overlaying="x2",
                    side="bottom",
                    minor_ticks="outside",
                    type="linear",
                    showspikes=True,
                    ),
                # hovermode="x unified",
                )

            # plot the modal profiles
            self.itrace_hz = self.itrace_epsilon_imag+1
            self.itraces_hz = list(range(self.itrace_hz, self.itrace_hz+self.nmodes_plot))
            for i in range(self.nmodes_plot):
                self.plotly_fig.add_trace(self.traces_hz[i], row=i+2, col=1)
                # self.plotly_fig.update_yaxes(title_text=r"$h_z$",title_standoff=0,range=[-1.5,1.1], row=i+2, col=1)
                self.plotly_fig.update_xaxes(title_text=r"$x$",title_standoff=0,row=i+2, col=1)
                # self.plotly_fig.add_annotation(x=0, y=1, yshift=0.1, text=r"$n_{\mathrm{eff}}$", showarrow=False, row=i+2, col=1)

            self.itrace_ez = self.itrace_hz+self.nmodes_plot
            self.itraces_ez = list(range(self.itrace_ez, self.itrace_ez+self.nmodes_plot))
            for i in range(self.nmodes_plot):
                self.plotly_fig.add_trace(self.traces_ez[i], row=i+2, col=2)
                # self.plotly_fig.update_yaxes(title_text=r"$e_z$",title_standoff=0,range=[-1.5,1.1], row=i+2, col=2)
                self.plotly_fig.update_xaxes(title_text=r"$x$",title_standoff=0,row=i+2, col=2)
                # self.plotly_fig.add_annotation(x=0, y=1, yshift=0.1, text=r"$n_{\mathrm{eff}}$", showarrow=False, row=i+2, col=2)

            self.traces_wgTE = [plotly_go.Scatter(
                x=self._model.region_wg,
                y=[1.1, 1.1],
                fill='tozeroy', fillcolor='rgba(180, 180, 180, 0.35)',
                mode='none',
                hoverinfo='skip',
                showlegend=False,
                ) for i in range(self.nmodes_plot)]

            self.traces_wgTM = [plotly_go.Scatter(
                x=self._model.region_wg,
                y=[1.1, 1.1],
                fill='tozeroy', fillcolor='rgba(180, 180, 180, 0.35)',
                mode='none',
                hoverinfo='skip',
                showlegend=False,
                ) for i in range(self.nmodes_plot)]

            # waveguide interval
            self.itrace_wgTE = self.itrace_ez + self.nmodes_plot
            self.itraces_wgTE = list(range(self.itrace_wgTE, self.itrace_wgTE+self.nmodes_plot))
            for imode in range(self.nmodes_plot):
                self.plotly_fig.add_trace(self.traces_wgTE[imode], row=imode+2, col=1)

            self.itrace_wgTM = self.itrace_wgTE + self.nmodes_plot
            self.itraces_wgTM = list(range(self.itrace_wgTM, self.itrace_wgTM+self.nmodes_plot))
            for imode in range(self.nmodes_plot):
                self.plotly_fig.add_trace(self.traces_wgTM[imode], row=imode+2, col=2)

            # draw intervals
            for imode in range(self.nmodes_plot):
                for col in range(2):
                    [self.plotly_fig.add_vline(x=interval, line_width=1, line_dash="dot", line_color="black", row=imode+2, col=col+1) for interval in self._model.geom.intervals]

                # layout_imode =
                self.plotly_fig.add_annotation(
                    x=self._model.geom.intervals[0], y=-1.5,
                    xanchor="left",
                    yanchor="bottom",
                    # bgcolor="white",
                    name="neffTM"+str(imode),
                    text="TM"+str(imode)+" ("+f"{0:.{self.digits}E}{0:+.{self.digits}E}j)",
                    showarrow=False,
                    font=dict(size=11),
                    row=imode+2, col=1,
                    )
                # print(layout_imode); sys.exit(0)

                self.plotly_fig.add_annotation(
                    x=self._model.geom.intervals[0], y=-1.5,
                    xanchor="left",
                    yanchor="bottom",
                    # bgcolor="white",
                    name="neffTE"+str(imode),
                    text="TE"+str(imode)+" ("+f"{0:.{self.digits}E}{0:+.{self.digits}E}j)",
                    showarrow=False,
                    font=dict(size=11),
                    row=imode+2, col=2,
                    )

            # for iw in range(self._model.wArray.size):
            # layout_iw = plotly_go.Layout(
            #     annotations=[
            #         plotly_go.layout.Annotation(
            #             x=self._model.geom.intervals[0],
            #             y=-1.5,
            #             xanchor="left",
            #             yanchor="bottom",
            #             # bgcolor="white",
            #             name="neffTM"+str(0),
            #             text="TM"+str(0)+" ("+f"{0:.{self.digits}E}{0:+.{self.digits}E}j)",
            #             showarrow=False,
            #             font=dict(size=11),
            #             row=0+2, col=1,
            #             )
            #         ],

            #     )

            # print(self.plotly_fig["layout"]["annotations"]); sys.exit(0)
            # frames for slider
            self.frames = [plotly_go.Frame(
                data = self.traces_TM+self.traces_TE+self.traces_hz+self.traces_ez+self.traces_wgTE+self.traces_wgTM+[self.trace_epsilon_real, self.trace_epsilon_imag],
                name = f"w{self._model.wArray[iw]:.2f}",
                traces = self.itraces_TM+self.itraces_TE+self.itraces_hz+self.itraces_ez+self.itraces_wgTE+self.itraces_wgTM+[self.itrace_epsilon_real,self.itrace_epsilon_imag],
                layout = self.plotly_fig["layout"],
                ) for iw in range(self._model.wArray.size)]

            self.plotly_fig.update(frames=self.frames)


            # animation parameters
            self.fr_duration=50
            self.sliders = [{
                "pad": {"b": 0, "t": 0},
                "len": 0.75,
                "x": 0.11,
                "y": (sum(pixels_row[1:self.nmodes_plot])+290)/pixels_fig,
                "steps": [
                    {
                    "args": [[f.name], self._frame_args(self.fr_duration)],
                    "label": f"w{self._model.wArray[iw]:.2f}",
                    "method": "animate",
                    }
                    for iw, f in enumerate(self.plotly_fig.frames)],
                }]
            self.updatemenus = [{
                "buttons": [
                    {
                      "args": [None, self._frame_args(self.fr_duration)],
                      "label": "&#9654;", # play symbol
                      "method": "animate",
                    },
                    {
                      "args": [[None], self._frame_args(self.fr_duration)],
                      "label": "&#9724;", # pause symbol
                      "method": "animate",
                    }],
                "active": -1,
                "direction": "left",
                "pad": {"r": 0, "t": 0},
                "type": "buttons",
                "x": 0.08,
                "y": (sum(pixels_row[1:self.nmodes_plot])+270)/pixels_fig,
                  }]
            if self._model.wArray.size>1:
                self.plotly_fig.update_layout(sliders=self.sliders, updatemenus=self.updatemenus)

        def Update(self, idx_TE, idx_TM, iw=0):

            ipx_wg = np.where( (self._model.mesh.px>=self._model.geom.region_wg[0]) & (self._model.mesh.px<=self._model.geom.region_wg[1]) )[0]
            ipx_phy = np.where( (self._model.mesh.px>=self._model.geom.region_phy[0]) & (self._model.mesh.px<=self._model.geom.region_phy[1]) )[0]
            ipx = np.where(self._model.mesh.px)[0]


            [ self.axs[1+id].clear() for id in range(self.nmodes_plot*2) ]
            # self.axs[-1+self.nmodes_plot*2].set_xlabel("x (m)")
            # self.axs[self.nmodes_plot*2].set_xlabel("x (m)")

            # sx = self._model._sx[:self._model.mesh.npoints] # scaling factor

            for id in np.arange(idx_TM.size):
                ic=np.mod(id, 10)

                self.axs[0].plot(self._model.sol.neff_TM[idx_TM[id]].real*self._model.k0, self._model.w, 'C'+str(ic)+'o', ms=6)

                hz = self._model.solArray[iw].hz_TM[idx_TM[id], :self._model.mesh.npoints]
                nx2 = (self._model._epsilon_raw-self._model.solArray[iw].neff_TM[idx_TM[id]]**2)
                # nx_re =
                nx = np.sqrt(nx2)
                kx = nx*self._model.k0
                n_raw = (self._model._epsilon_raw*self._model._mu_raw)**0.5

                if id<self.nmodes_plot:

                    self.axs[1+id*2].plot(self._model.mesh.px[ipx], np.real(hz[ipx]), 'C'+str(ic)+'-o', lw=1, label='TM'+str(id), ms=2);
                    # self.axs[1+id*2].plot(self._model.mesh.px[ipx], np.real(sx[ipx]), 'k-', lw=0.5, label='TM'+str(id));
                    # self.axs[1+id*2].plot(self._model.mesh.px[ipx], np.imag(sx[ipx]), 'k--', lw=0.5, label='TM'+str(id));
                    self.axs[1+id*2].plot(self._model.mesh.px[ipx], np.real(nx[ipx]), 'k-', lw=0.5, label='TM'+str(id));
                    self.axs[1+id*2].plot(self._model.mesh.px[ipx], np.imag(nx[ipx]), 'k--', lw=0.5, label='TM'+str(id));
                    self.axs[1+id*2].plot(self._model.mesh.px[ipx], np.real(n_raw[ipx]), 'b-', lw=0.5, label='TM'+str(id));
                    self.axs[1+id*2].plot(self._model.mesh.px[ipx], np.imag(n_raw[ipx]), 'b--', lw=0.5, label='TM'+str(id));


                    self.axs[1+id*2].fill_between(self._model.geom.region_wg, (-1, -1), (1, 1), color='gray', alpha=0.2)
                    self.axs[1+id*2].vlines(np.array(self._model.geom.intervals), ymin=-1, ymax=1, colors="k", linestyles='dotted')

                    # self.axs[1+id*2].set_ylim([-1.2, 1.2])
                    self.axs[1+id*2].set_xlabel("x (m)")

                    if self.debugging:
                        self.axs[1+id*2].annotate(f"{self._model.solArray[iw].neff_TM[idx_TM[id]]:+.{self.digits}E}", (1.0, 0.9), xycoords='axes fraction', va="top", ha="right")
                        self.axs[1+id*2].annotate("TM$_{"+str(id)+"}$ ($\\kappa$="+f"{self._model.solArray[iw].kappa_Syleft_TM[idx_TM[id]]:.{self.digits}E})", (0.02, 0.95), xycoords='axes fraction', va="top", ha="left")
                        self.axs[1+id*2].annotate("TM$_{"+str(id)+"}$ (    "+f"{self._model.solArray[iw].kappa_Syright_TM[idx_TM[id]]:.{self.digits}E})", (0.02, 0.50), xycoords='axes fraction', va="center", ha="left")

            for id in np.arange(idx_TE.size):
                ic=np.mod(id, 10)

                self.axs[0].plot(self._model.solArray[iw].neff_TE[idx_TE[id]].real*self._model.k0, self._model.w, 'C'+str(ic)+'s', ms=6)

                ez = self._model.solArray[iw].ez_TE[idx_TE[id], :self._model.mesh.npoints]
                nx2 = (self._model._epsilon_raw-self._model.solArray[iw].neff_TM[idx_TM[id]]**2)
                nx = np.sqrt(nx2)
                kx = nx*self._model.k0
                n_raw = (self._model._epsilon_raw*self._model._mu_raw)**0.5

                if id<self.nmodes_plot:

                    self.axs[2+id*2].plot(self._model.mesh.px[ipx], np.real(ez[ipx]), 'C'+str(ic)+'--o', lw=1, label='TE'+str(id), ms=2);
                    # self.axs[2+id*2].plot(self._model.mesh.px[ipx], np.real(sx[ipx]), 'k-', lw=0.5, label='TE'+str(id));
                    # self.axs[2+id*2].plot(self._model.mesh.px[ipx], np.imag(sx[ipx]), 'k--', lw=0.5, label='TE'+str(id));
                    self.axs[2+id*2].plot(self._model.mesh.px[ipx], np.real(nx[ipx]), 'k-', lw=0.5, label='TE'+str(id));
                    self.axs[2+id*2].plot(self._model.mesh.px[ipx], np.imag(nx[ipx]), 'k--', lw=0.5, label='TE'+str(id));
                    self.axs[2+id*2].plot(self._model.mesh.px[ipx], np.real(n_raw[ipx]), 'b-', lw=0.5, label='TE'+str(id));
                    self.axs[2+id*2].plot(self._model.mesh.px[ipx], np.imag(n_raw[ipx]), 'b--', lw=0.5, label='TE'+str(id));

                    self.axs[2+id*2].fill_between(self._model.geom.region_wg, (-1, -1), (1, 1), color='gray', alpha=0.2)
                    self.axs[2+id*2].vlines(np.array(self._model.geom.intervals), ymin=-1, ymax=1, colors="k", linestyles='dotted')

                    # self.axs[2+id*2].set_ylim([-1.2, 1.2])
                    self.axs[2+id*2].set_xlabel("x (m)")

                    if self.debugging:
                        self.axs[2+id*2].annotate(f"{self._model.solArray[iw].neff_TE[idx_TE[id]]:+.{self.digits}E}", (1.0, 0.9), xycoords='axes fraction', va="top", ha="right")
                        self.axs[2+id*2].annotate("TE$_{"+str(id)+"}$ ($\\kappa$="+f"{self._model.solArray[iw].kappa_Syleft_TE[idx_TE[id]]:.{self.digits}E})", (0.02, 0.95), xycoords='axes fraction', va="top", ha="left")
                        self.axs[2+id*2].annotate("TE$_{"+str(id)+"}$ (    "+f"{self._model.solArray[iw].kappa_Syright_TE[idx_TE[id]]:.{self.digits}E})", (0.02, 0.50), xycoords='axes fraction', va="center", ha="left")

            self.fig.canvas.draw();
            self.fig.canvas.flush_events(); # update the plot without any pause

        def Update_plotly(self, idx_TE, idx_TM, iw=0):
            # update modal dispersion curves
            if np.size(idx_TM)>0:
                for i, i_raw in enumerate(idx_TM):
                    if i<self.nmodes_plot:
                        # update trace data in dispersion curves
                        self.neff_TM[iw, i] = self._model.sol.neff_TM[i_raw]
                        self.frames[iw].data[i].x = self.neff_TM[:, i].real*2*np.pi/self._model.ld0Array # slider plot
                        # self.frames[iw].data[i].name = "TM"+str(i)+" ("+f"{self.neff_TM[iw, i]:+.{self.digits}E})"
                        self.frames[iw].layout.annotations[2*i].text = "TM"+str(i)+" ("+f"{self.neff_TM[iw, i]:+.{self.digits}E})"
                        # self.plotly_fig.update_annotations(
                        #     selector={"name":"neffTM"+str(i)},
                        #     text="TM"+str(i)+" ("+f"{self.neff_TM[iw, i]:+.{self.digits}E})",
                        #     )

                        # update the modal profile
                        self.frames[iw].data[2*self.nmodes_plot+i].y = self._model.sol.hz_TM[i_raw, :self._model.mesh.npoints].real
                        # self.frames[iw].data[2*self.nmodes_plot+i].y = self._model.sol.Sx_TM[i_raw, :self._model.mesh.npoints].real
                        # imax_Sx_TM = np.nanargmax(np.real(self._model.sol.Sx_TM[i_raw, :self._model.mesh.npoints]))
                        # self.frames[iw].data[2*self.nmodes_plot+i].y = (self._model.sol.Sx_TM[i_raw, :self._model.mesh.npoints]/self._model.sol.Sx_TM[i_raw, imax_Sx_TM]).real
                        # print(i_raw)

            # update modal dispersion curves
            if np.size(idx_TE)>0:
                for i, i_raw in enumerate(idx_TE):
                    if i<self.nmodes_plot:
                        self.neff_TE[iw, i] = self._model.sol.neff_TE[i_raw]
                        self.frames[iw].data[self.nmodes_plot+i].x = self.neff_TE[:, i].real*2*np.pi/self._model.ld0Array # slider plot
                        # self.frames[iw].data[self.nmodes_plot+i].name = "TE"+str(i)+" ("+f"{self.neff_TE[iw, i]:+.{self.digits}E})"

                        self.frames[iw].layout.annotations[2*i+1].text = "TE"+str(i)+" ("+f"{self.neff_TE[iw, i]:+.{self.digits}E})"
                        # self.plotly_fig.update_annotations(
                        #     selector={"name":"neffTE"+str(i)},
                        #     text="TE"+str(i)+" ("+f"{self.neff_TE[iw, i]:+.{self.digits}E})",
                        #     )

                        # update the modal profile
                        self.frames[iw].data[3*self.nmodes_plot+i].y = self._model.sol.ez_TE[i_raw, :self._model.mesh.npoints].real
                        # self.frames[iw].data[3*self.nmodes_plot+i].y = self._model.sol.Sx_TE[i_raw, :self._model.mesh.npoints].real
                        # imax_Sx_TE = np.nanargmax(np.real(self._model.sol.Sx_TE[i_raw, :self._model.mesh.npoints]))
                        # self.frames[iw].data[3*self.nmodes_plot+i].y = (self._model.sol.Sx_TE[i_raw, :self._model.mesh.npoints]/self._model.sol.Sx_TE[i_raw, imax_Sx_TE]).real

            for imode in range(self.nmodes_plot):
                self.frames[iw].data[4*self.nmodes_plot+imode].x = self._model.region_wg
                self.frames[iw].data[5*self.nmodes_plot+imode].x = self._model.region_wg

            self.plotly_fig.update(frames=self.frames)

        def Save(self, figname="plotly"):
            self.plotly_fig.write_html(
                figname+".html",
                include_plotlyjs="../../js/plotly-2.27.0.min.js",
                include_mathjax="../../js/MathJax-2.7.7/MathJax.js",
            )

    def _ModeInnerProduct(self, eigvec1, eigvec2, mesh=None, order=4):
        dot = 0
        # by direct summation
        dot = np.sum(eigvec1[:self.mesh.npoints]*np.conj(eigvec2[:self.mesh.npoints]))

        # by intergration
        # mode1_profile = ngs.GridFunction(self.fes)
        # mode2_profile = ngs.GridFunction(self.fes)

        # mode1_profile.vec.data = eigvec1
        # mode2_profile.vec.data = eigvec2

        # if mesh==None:
        #     dot = ngs.Integrate( mode1_profile*ngs.Conj(mode2_profile), self.mesh._ngsolve_mesh, order=order)
        # else:
        #     dot = ngs.Integrate( mode1_profile*ngs.Conj(mode2_profile), mesh, order=order)

        return dot

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

                # if type(pml_epsilon_raw)==ngs.CoefficientFunction:
                #     pml_epsilon_raw = (pml_epsilon_raw(self.mesh._ngsolve_mesh(self.mesh.cpoint))[0])[0]

                # if type(pml_mu_raw)==ngs.CoefficientFunction:
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
                # pml_sigma0_max = -np.log(pml_R0)*(pml_mPoly+1)/(const_eta_0*tPML*2)
                pml_sigma0_max = 0.8*(pml_mPoly+1)/(const_eta_0*pml_deltax*ngs.sqrt(pml_epsilon_raw*pml_mu_raw)) # optimal one

                # alpha_max=0.24 is used in the reference:
                # "S. D. Gedney, Scaled CFS-PML: It Is More Robust, More Accurate,
                # More Efficient, and Simple to Implement. Why Aren’t You Using
                # It?, in 2005 IEEE Antennas and Propagation Society International
                # Symposium, Vol. 4B (IEEE, Washington, DC, USA, 2005), pp. 364–367."
                pml_alpha_max = 0.24 # for fulfilling the causality

                pml_kappa_max = pml_sigma0_max/(self._ngsolve_omega*const_epsilon_0) # for estimation at the current wavelength

                pml_sigma_max = pml_sigma0_max/ngs.sqrt(pml_epsilon_raw*pml_mu_raw)

                # ngs.x is the coordinate variable provided by ngsolve
                print(pml_cpoint-self.mesh.cpoint[0])
                pml_delta = ngs.IfPos(pml_cpoint-self.mesh.cpoint[0], ngs.x-pml_x_left, pml_x_right-ngs.x)

                #### kappa, sigma, alpha
                # imag(kx)*kappa is integrated to zero, when kappa = 1-2*(pml_delta/tPML)**pml_mPoly
                # pml_kappa_max = -1 # the integral approaches to zero
                pml_kappa_max = 0 # the scaling function approaches to zero
                # pml_kappa = 1 + (pml_kappa_max-1)*(pml_delta/tPML)**pml_mPoly
                pml_kappa = 1 + (pml_kappa_max-1)*(pml_delta/tPML)**(1/pml_mPoly) # inverse polynomial
                # pml_kappa = 0.5 -0.5*(ngs.atan((pml_delta - tPML/2)/(tPML/5))/(np.pi/2)/(np.arctan(5/2)/(np.pi/2)))

                pml_sigma = pml_sigma_max*(pml_delta/tPML)**pml_mPoly
                pml_alpha = pml_alpha_max*((tPML-pml_delta)/tPML)**pml_mPoly

                # Under the convention exp(-j\omega t), it is "+1j" in the denominator
                # in "S. D. Gedney, Introduction to the Finite-Difference Time-Domain (FDTD)
                # Method for Electromagnetics (Springer International Publishing, Cham, 2011)"
                # Here, we stick with the convention exp(i\omega t),
                # so we use "-1j" in the denominator.
                pml_sx = pml_kappa + pml_sigma/(pml_alpha*0-1j*self._ngsolve_omega*const_epsilon_0)
                # multiplying pml_alpha by 0 to disable that term for studying leaky modes
                # pml_sx = pml_kappa + 5j*(pml_delta/tPML)**pml_mPoly # a test scaling function

                if pml_plot:
                    pml_xArray = np.linspace(pml_x_left, pml_x_right, nPML)
                    pml_sxArray = np.full_like(pml_xArray, np.nan, dtype=np.complex64)
                    pml_epsilon_rawArray = np.full_like(pml_xArray, np.nan, dtype=np.complex64)
                    pml_mu_rawArray = np.full_like(pml_xArray, np.nan, dtype=np.complex64)

                    for ix in np.arange(pml_xArray.size):
                        pml_sxArray[ix] = pml_sx(self.mesh._ngsolve_mesh(pml_xArray[ix]))
                        if type(pml_epsilon_raw)==ngs.CoefficientFunction:
                            pml_epsilon_rawArray[ix] = pml_epsilon_raw(self.mesh._ngsolve_mesh(pml_xArray[ix]))
                        else:
                            pml_epsilon_rawArray[ix] = pml_epsilon_raw
                        if type(pml_mu_raw)==ngs.CoefficientFunction:
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

                #### Store PML information
                # in a dictionary with label as the key, and new epsilon, mu and old epsilon, mu
                # self._PMLs |= {pml_dom: [pml_epsilon_xx, pml_epsilon_yy, pml_epsilon_zz, pml_mu_xx, pml_mu_yy, pml_mu_zz, self.material._map_epsilon[pml_dom], self.material._map_mu[pml_dom]]}
                self._PMLs |= {pml_dom: pml_sx}

                #### Update PML domains with new scaling function
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

    #### setTBC() accepts two numbers defining the indices of the leftmost to the
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
            if not type(self.material._map_epsilon[label])==float:
                map_epsilon_re[label] = self.material._map_epsilon[label].real
                map_epsilon_im[label] = self.material._map_epsilon[label].imag
            else:
                map_epsilon_im[label] = 0

        self._cf_epsilon_raw_re = self.mesh._ngsolve_mesh.MaterialCF(map_epsilon_re, default=self.material.default_epsilon)
        self._cf_epsilon_raw_im = self.mesh._ngsolve_mesh.MaterialCF(map_epsilon_im, default=0)
        self._cf_epsilon_raw = self._cf_epsilon_raw_re + 1j*self._cf_epsilon_raw_im
        # real and imaginary parts
        # self._cf_epsilon_raw_re = 0.5*(self._cf_epsilon_raw + ngs.Conj(self._cf_epsilon_raw))
        # self._cf_epsilon_raw_im = -0.5j*(self._cf_epsilon_raw - ngs.Conj(self._cf_epsilon_raw))

        map_mu_re = self.material._map_mu.copy()
        map_mu_im = self.material._map_mu.copy()
        for i, label in enumerate(self.material._map_mu):
            if not type(self.material._map_mu[label])==float:
                map_mu_re[label] = self.material._map_mu[label].real
                map_mu_im[label] = self.material._map_mu[label].imag
            else:
                map_mu_im[label] = 0

        self._cf_mu_raw_re = self.mesh._ngsolve_mesh.MaterialCF(map_mu_re, default=self.material.default_mu)
        self._cf_mu_raw_im = self.mesh._ngsolve_mesh.MaterialCF(map_mu_im, default=0)
        self._cf_mu_raw = self._cf_mu_raw_re + 1j*self._cf_mu_raw_im
        # real and imaginary parts
        # self._cf_mu_raw_re = 0.5*(self._cf_mu_raw + ngs.Conj(self._cf_mu_raw))
        # self._cf_mu_raw_im = -0.5j*(self._cf_mu_raw - ngs.Conj(self._cf_mu_raw))

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
        self._cf_epsilon_xx = self._cf_epsilon_raw/self._cf_sx
        self._cf_epsilon_yy = self._cf_epsilon_raw*self._cf_sx
        self._cf_epsilon_zz = self._cf_epsilon_yy
        # 3 diagonal elements of the mu tensor
        self._cf_mu_xx = self._cf_mu_raw/self._cf_sx
        self._cf_mu_yy = self._cf_mu_raw*self._cf_sx
        self._cf_mu_zz = self._cf_mu_yy

        # real and imaginary parts
        self._cf_epsilon_xx_re = 0.5*(self._cf_epsilon_xx + ngs.Conj(self._cf_epsilon_xx))
        self._cf_epsilon_xx_im = -0.5j*(self._cf_epsilon_xx - ngs.Conj(self._cf_epsilon_xx))
        self._cf_epsilon_yy_re = 0.5*(self._cf_epsilon_yy + ngs.Conj(self._cf_epsilon_yy))
        self._cf_epsilon_yy_im = -0.5j*(self._cf_epsilon_yy - ngs.Conj(self._cf_epsilon_yy))
        self._cf_epsilon_zz_re = self._cf_epsilon_yy_re
        self._cf_epsilon_zz_im = self._cf_epsilon_yy_im

        self._cf_mu_xx_re = 0.5*(self._cf_mu_xx + ngs.Conj(self._cf_mu_xx))
        self._cf_mu_xx_im = -0.5j*(self._cf_mu_xx - ngs.Conj(self._cf_mu_xx))
        self._cf_mu_yy_re = 0.5*(self._cf_mu_yy + ngs.Conj(self._cf_mu_yy))
        self._cf_mu_yy_im = -0.5j*(self._cf_mu_yy - ngs.Conj(self._cf_mu_yy))
        self._cf_mu_zz_re = self._cf_mu_yy_re
        self._cf_mu_zz_im = self._cf_mu_yy_im

        # dispersive properties
        self._cf_epsilon_xx_domega = self._cf_epsilon_xx.Diff(self._ngsolve_omega)
        self._cf_epsilon_yy_domega = self._cf_epsilon_yy.Diff(self._ngsolve_omega)
        self._cf_epsilon_zz_domega = self._cf_epsilon_yy_domega
        self._cf_mu_xx_domega = self._cf_mu_xx.Diff(self._ngsolve_omega)
        self._cf_mu_yy_domega = self._cf_mu_yy.Diff(self._ngsolve_omega)
        self._cf_mu_zz_domega = self._cf_mu_yy_domega

        #### with TBC
        if self.TBC:
            self._A0TM = ngs.BilinearForm(self.fes)
            self._A1TM = ngs.BilinearForm(self.fes, check_unused=False) # matrix for the boundary integral
            self._A2TM = ngs.BilinearForm(self.fes)
            self._A3TM = ngs.BilinearForm(self.fes, check_unused=False) # matrix for the boundary integral
            self._A4TM = ngs.BilinearForm(self.fes)

            self._A0TE = ngs.BilinearForm(self.fes)
            self._A1TE = ngs.BilinearForm(self.fes, check_unused=False) # matrix for the boundary integral
            self._A2TE = ngs.BilinearForm(self.fes)
            self._A3TE = ngs.BilinearForm(self.fes, check_unused=False) # matrix for the boundary integral
            self._A4TE = ngs.BilinearForm(self.fes)

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

            # self._slepc_PEP = False
            self._slepc_PEP = True
            self._slepc_NEP = not self._slepc_PEP

            #### formulated as a polynomial nonlinear eigenvalue problem
            if self._slepc_PEP:

                # TM mode
                self._A0TM += self._delta2**2*self._ngsolve_k0**2*1/(self._cf_epsilon_xx)*self._v*self._u*ngs.dx(definedon=self.mesh._wg)
                self._A1TM += 4j*(
                    self._ngsolve_k0*self._delta2/(self._epsilon_yy_right)*self._v.Trace()*self._u.Trace()
                    *ngs.ds(definedon=self.mesh._bnd_right)
                    -1*self._ngsolve_k0*self._delta2/(self._epsilon_yy_left)*self._v.Trace()*self._u.Trace()
                    *ngs.ds(definedon=self.mesh._bnd_left))
                self._A2TM += (
                    -16*(1/(self._cf_epsilon_yy)*ngs.grad(self._v)*ngs.grad(self._u)*ngs.dx(definedon=self.mesh._wg) - self._ngsolve_k0**2*self._cf_mu_zz*self._v*self._u*ngs.dx(definedon=self.mesh._wg))
                    -8*self._Sigma2*self._ngsolve_k0**2*1/self._cf_epsilon_xx*self._v*self._u*ngs.dx(definedon=self.mesh._wg))
                self._A3TM += 16j*(
                    self._ngsolve_k0/(self._epsilon_yy_right)*self._v.Trace()*self._u.Trace()
                    *ngs.ds(definedon=self.mesh._bnd_right)
                    +self._ngsolve_k0/(self._epsilon_yy_left)*self._v.Trace()*self._u.Trace()
                    *ngs.ds(definedon=self.mesh._bnd_left))
                self._A4TM += 16*self._ngsolve_k0**2*1/(self._cf_epsilon_xx)*self._v*self._u*ngs.dx(definedon=self.mesh._wg)

                # TE mode
                self._A0TE += self._delta2**2*self._ngsolve_k0**2*1/(self._cf_mu_xx)*self._v*self._u*ngs.dx(definedon=self.mesh._wg)
                self._A1TE += 4j*(
                    self._ngsolve_k0*self._delta2/(self._mu_yy_right)*self._v.Trace()*self._u.Trace()
                    *ngs.ds(definedon=self.mesh._bnd_right)
                    -1*self._ngsolve_k0*self._delta2/(self._mu_yy_left)*self._v.Trace()*self._u.Trace()
                    *ngs.ds(definedon=self.mesh._bnd_left))
                self._A2TE += (
                    -16*(1/(self._cf_mu_yy)*ngs.grad(self._v)*ngs.grad(self._u)*ngs.dx(definedon=self.mesh._wg) - self._ngsolve_k0**2*self._cf_epsilon_zz*self._v*self._u*ngs.dx(definedon=self.mesh._wg))
                    -8*self._Sigma2*self._ngsolve_k0**2*1/self._cf_mu_xx*self._v*self._u*ngs.dx(definedon=self.mesh._wg))
                self._A3TE += 16j*(
                    self._ngsolve_k0/(self._mu_yy_right)*self._v.Trace()*self._u.Trace()
                    *ngs.ds(definedon=self.mesh._bnd_right)
                    +self._ngsolve_k0/(self._mu_yy_left)*self._v.Trace()*self._u.Trace()
                    *ngs.ds(definedon=self.mesh._bnd_left))
                self._A4TE += 16*self._ngsolve_k0**2*1/(self._cf_mu_xx)*self._v*self._u*ngs.dx(definedon=self.mesh._wg)

                self._AlistTM = [self._A0TM, self._A1TM, self._A2TM, self._A3TM, self._A4TM]
                self._AlistTE = [self._A0TE, self._A1TE, self._A2TE, self._A3TE, self._A4TE]

            #FIXME:
            #### formulated as a general split form nonlinear eigenvalue problem
            # when self._slepc_NEP == True
            else:
                pass
                # self._A0TM += (1/(self._cf_epsilon_yy)
                #                *ngs.grad(self._v)*ngs.grad(self._u)*ngs.dx(definedon=self.mesh._wg)
                #                -self._ngsolve_k0**2*self._cf_mu_zz
                #                *self._v*self._u*ngs.dx(definedon=self.mesh._wg))
                # self._A1TM += 1/(self._cf_epsilon_xx)*self._v*self._u*ngs.dx(definedon=self.mesh._wg)
                # self._A2TM += -1j/(self._cf_epsilon_yy)*self._v.Trace()*self._u.Trace()*ngs.ds(definedon=self.mesh._bnd_right)
                # self._A3TM += -1j/(self._cf_epsilon_yy)*self._v.Trace()*self._u.Trace()*ngs.ds(definedon=self.mesh._bnd_left)

                # self._A0TE += (1/(self._cf_mu_yy)
                #                *ngs.grad(self._v)*ngs.grad(self._u)*ngs.dx(definedon=self.mesh._wg)
                #                -self._ngsolve_k0**2*self._cf_epsilon_zz
                #                *self._v*self._u*ngs.dx(definedon=self.mesh._wg))
                # self._A1TE += 1/(self._cf_mu_xx)*self._v*self._u*ngs.dx(definedon=self.mesh._wg)
                # self._A2TE += -1j/(self._cf_mu_yy)*self._v.Trace()*self._u.Trace()*ngs.ds(definedon=self.mesh._bnd_right)
                # self._A3TE += -1j/(self._cf_mu_yy)*self._v.Trace()*self._u.Trace()*ngs.ds(definedon=self.mesh._bnd_left)

                # self._AlistTM = [self._A0TM, self._A1TM, self._A2TM, self._A3TM]
                # self._AlistTE = [self._A0TE, self._A1TE, self._A2TE, self._A3TE]

        #### w/o TBC: PML backed by PEC
        else:
            # Weak formulation
            self._A0TM = ngs.BilinearForm(self.fes)
            self._mTM = ngs.BilinearForm(self.fes)
            self._A0TE = ngs.BilinearForm(self.fes)
            self._mTE = ngs.BilinearForm(self.fes)

            # k0 = omega0/c_const
            # TM mode
            self._A0TM += 1/(self._cf_epsilon_yy)*ngs.grad(self._v)*ngs.grad(self._u)*ngs.dx- self._ngsolve_k0**2*self._cf_mu_zz*self._v*self._u*ngs.dx
            self._mTM += -1*self._ngsolve_k0**2/self._cf_epsilon_xx*self._v*self._u*ngs.dx
            # TE mode
            self._A0TE += 1/(self._cf_mu_yy)*ngs.grad(self._v)*ngs.grad(self._u)*ngs.dx- self._ngsolve_k0**2*self._cf_epsilon_zz*self._v*self._u*ngs.dx
            self._mTE += -1*self._ngsolve_k0**2/self._cf_mu_xx*self._v*self._u*ngs.dx

        ## number of eigenvalues to be sought
        # self.sol.neigs = self.fes.ndof # all eigen values
        #### TODO: Solve eigenvalues less than "ndof".
        # self.sol.neigs = np.minimum(20, self.fes.ndof)

    # 1. Apply mode filter first
    # 2. Sort filtered modes (less and faster)
    def Sort(self, normalization=False):

        print("Sorting modes at the normalized frequency w = "+"{0.real:.3f}".format(self.w))

        # self._n_raw = np.sqrt(self._epsilon_raw*self._mu_raw)
        # self._n = np.sqrt(self._epsilon*self._mu)

        # self._n_raw_max = np.max(np.abs(self._n_raw))
        # self._n_raw_real_min = np.min(np.abs(self._n_raw.real))
        # self._epsilon_raw_max = np.max(np.abs(self._epsilon_raw))
        # self._epsilon_raw_real_min = np.min(np.abs(self._epsilon_raw.real))

        # 3 diagonal elements
        # self._n_max = np.max(np.abs(self._n), axis=0)
        # self._n_real_min = np.min(np.abs(self._n.real), axis=0)
        # self._epsilon_real_min = np.min(np.abs(self._epsilon.real), axis=0)
        # return
        self._epsilon_max = np.max(np.abs(self._epsilon), axis=0)

        #### auxiliary coefficient functions
        # stepfun_wg: a step function representing the waveguiding region
        # stepfun_phy: a step function representing the physical region
        # (whole computation region excluding PMLs)
        stepfun_wg = ngs.IfPos(-ngs.x+self.geom.region_wg[0], 0, 1) * ngs.IfPos(ngs.x-(+self.geom.region_wg[1]), 0, 1)
        stepfun_phy = ngs.IfPos(-ngs.x+self.geom.region_phy[0], 0, 1) * ngs.IfPos(ngs.x-(+self.geom.region_phy[1]), 0, 1)

        stepfun_pml_left = ngs.IfPos(-ngs.x+self.geom.intervals[0], 0, 1) * ngs.IfPos(ngs.x-(+self.geom.intervals[1]), 0, 1)
        stepfun_pml_right = ngs.IfPos(-ngs.x+self.geom.intervals[-2], 0, 1) * ngs.IfPos(ngs.x-(+self.geom.intervals[-1]), 0, 1)

        # stepfun_clad_left = ngs.IfPos(-ngs.x+self.geom.intervals[1], 0, 1) * ngs.IfPos(ngs.x-(+self.geom.region_wg[0]), 0, 1)
        # stepfun_clad_right = ngs.IfPos(-ngs.x+self.geom.region_wg[1], 0, 1) * ngs.IfPos(ngs.x-(+self.geom.intervals[-2]), 0, 1)

        # whether of not taking into account the dispersive term in energy densities
        dispersiveFactor = 1 if self.sol.dispersive else 0

        # beta = ngs.Parameter(1.0*self.k0)
        beta_re = ngs.Parameter(1.0*self.k0)
        beta_im = ngs.Parameter(1.0*self.k0)

        # For working on the ith mode profile
        hz = ngs.GridFunction(self.fes)
        ex = -(beta_re+1j*beta_im)/(self._ngsolve_omega*const_epsilon_0*self._cf_epsilon_xx)*hz
        ey = -1j/(self._ngsolve_omega*const_epsilon_0*self._cf_epsilon_yy)*hz.Deriv()

        ez = ngs.GridFunction(self.fes)
        hx = (beta_re+1j*beta_im)/(self._ngsolve_omega*const_mu_0*self._cf_mu_xx)*ez
        hy = 1j/(self._ngsolve_omega*const_mu_0*self._cf_mu_yy)*ez.Deriv()

        #### TM: auxiliary coefficient functions
        # beta*ngs.Conj(beta) = (beta_re**2+beta_im**2)
        # TM: the energy dissipation profiles
        Qe_TM = 1/(2*self._ngsolve_omega*const_epsilon_0)*(
            self._cf_epsilon_xx_im*(beta_re**2+beta_im**2)/(
                self._cf_epsilon_xx*ngs.Conj(self._cf_epsilon_xx)
                )*hz*ngs.Conj(hz)
            +self._cf_epsilon_yy_im/(
                self._cf_epsilon_yy*ngs.Conj(self._cf_epsilon_yy)
                )*hz.Deriv()*ngs.Conj(hz.Deriv())
            )

        Qm_TM = 0.5*self._ngsolve_omega*const_mu_0*(
            self._cf_mu_zz_im*hz*ngs.Conj(hz))

        # TM: the energy density profiles
        # if ... else ... can reduce the lenth of the
        # coefficient function in the case of dispersionless
        if self.sol.dispersive:
            ue_TM = 0.25/(self._ngsolve_omega**2*const_epsilon_0)*(
                (self._cf_epsilon_xx_re
                  +dispersiveFactor*self._ngsolve_omega*0.5*(
                    self._cf_epsilon_xx_domega+ngs.Conj(self._cf_epsilon_xx_domega))
                )*((beta_re**2+beta_im**2)/(
                    self._cf_epsilon_xx*ngs.Conj(self._cf_epsilon_xx))
                )*hz*ngs.Conj(hz)
                +(self._cf_epsilon_yy_re
                  +dispersiveFactor*self._ngsolve_omega*0.5*(
                    self._cf_epsilon_yy_domega+ngs.Conj(self._cf_epsilon_yy_domega))
                )/(self._cf_epsilon_yy*ngs.Conj(self._cf_epsilon_yy)
                )*hz.Deriv()*ngs.Conj(hz.Deriv())
            )
            um_TM = 0.25*const_mu_0*(
                self._cf_mu_zz_re
                +dispersiveFactor*self._ngsolve_omega*0.5*(
                    self._cf_mu_zz_domega+ngs.Conj(self._cf_mu_zz_domega))
                )*hz*ngs.Conj(hz)
        else:
            ue_TM = 0.25/(self._ngsolve_omega**2*const_epsilon_0)*(
                (self._cf_epsilon_xx_re
                 )*((beta_re**2+beta_im**2)/(
                    self._cf_epsilon_xx*ngs.Conj(self._cf_epsilon_xx))
                )*hz*ngs.Conj(hz)
                +(self._cf_epsilon_yy_re
                  )/(self._cf_epsilon_yy*ngs.Conj(self._cf_epsilon_yy)
                )*hz.Deriv()*ngs.Conj(hz.Deriv())
            )
            um_TM = 0.25*const_mu_0*(self._cf_mu_zz_re)*hz*ngs.Conj(hz)

        # TM: Mode confinement factor is the ratio of the integration of a quantity over two different regions.
        udiff2_TM = ngs.sqrt((ue_TM-um_TM)*ngs.Conj(ue_TM-um_TM)) # |ue_TM-um_TM|^2
        u_TM = (ue_TM+um_TM) # (ue_TM+um_TM)

        # TM: the power flow
        # beta = (beta_re + 1j*beta_im)
        Sy_TM = 1/(2*self._ngsolve_omega*const_epsilon_0)*0.5*(
            (beta_re + 1j*beta_im)*self._cf_sx/self._cf_epsilon_raw
            +ngs.Conj((beta_re + 1j*beta_im)*self._cf_sx/self._cf_epsilon_raw)
            )*hz*ngs.Conj(hz)

        Sx_TM = -1j/(2*self._ngsolve_omega*const_epsilon_0)*0.5*(
            hz.Deriv()*ngs.Conj(hz)
                /(self._cf_sx*self._cf_epsilon_raw)
            -ngs.Conj(hz.Deriv()*ngs.Conj(hz)
                /(self._cf_sx*self._cf_epsilon_raw))
            )

        #### TE: the energy dissipation profiles
        Qe_TE = 0.5*self._ngsolve_omega*const_epsilon_0*(
            self._cf_epsilon_zz_im*ez*ngs.Conj(ez))

        Qm_TE = 1/(2*self._ngsolve_omega*const_mu_0)*(
            self._cf_mu_xx_im*(beta_re**2+beta_im**2)/(
                self._cf_mu_xx*ngs.Conj(self._cf_mu_xx)
                )*ez*ngs.Conj(ez)
            +self._cf_mu_yy_im/(
                self._cf_mu_yy*ngs.Conj(self._cf_mu_yy)
                )*ez.Deriv()*ngs.Conj(ez.Deriv())
            )

        #### TE: auxiliary coefficient functions
        # TE: the energy density profiles
        if self.sol.dispersive:
            ue_TE = 0.25*const_epsilon_0*(
                self._cf_epsilon_zz_re
                +dispersiveFactor*self._ngsolve_omega*0.5*(
                    self._cf_epsilon_zz_domega
                    +ngs.Conj(self._cf_epsilon_zz_domega)
                    )
                )*ez*ngs.Conj(ez)

            um_TE = 0.25/(self._ngsolve_omega**2*const_mu_0)*(
                (self._cf_mu_xx_re
                  +dispersiveFactor*self._ngsolve_omega*0.5*(
                    self._cf_mu_xx_domega+ngs.Conj(self._cf_mu_xx_domega))
                )*((beta_re**2+beta_im**2)/(
                    self._cf_mu_xx*ngs.Conj(self._cf_mu_xx))
                )*ez*ngs.Conj(ez)
                +(self._cf_mu_yy_re
                  +dispersiveFactor*self._ngsolve_omega*0.5*(
                    self._cf_mu_yy_domega+ngs.Conj(self._cf_mu_yy_domega))
                )/(self._cf_mu_yy*ngs.Conj(self._cf_mu_yy)
                )*ez.Deriv()*ngs.Conj(ez.Deriv())
            )
        else:
            ue_TE = 0.25*const_epsilon_0*(
                self._cf_epsilon_zz_re
                )*ez*ngs.Conj(ez)
            um_TE = 0.25/(self._ngsolve_omega**2*const_mu_0)*(
                (self._cf_mu_xx_re
                )*((beta_re**2+beta_im**2)/(
                    self._cf_mu_xx*ngs.Conj(self._cf_mu_xx))
                )*ez*ngs.Conj(ez)
                +(self._cf_mu_yy_re
                )/(self._cf_mu_yy*ngs.Conj(self._cf_mu_yy)
                )*ez.Deriv()*ngs.Conj(ez.Deriv())
            )

        # Mode confinement factor is the ratio of the integration of a quantity over two different regions.
        udiff2_TE = ngs.sqrt((ue_TE-um_TE)*ngs.Conj(ue_TE-um_TE)) # |ue_TE-um_TE|
        u_TE = (ue_TE+um_TE) # (ue_TE+um_TE)

        # TE: the power flow
        Sy_TE =1/(2*self._ngsolve_omega*const_mu_0)*0.5*(
            (beta_re + 1j*beta_im)*self._cf_sx/self._cf_mu_raw
            +ngs.Conj((beta_re + 1j*beta_im)*self._cf_sx/self._cf_mu_raw)
            )*ez*ngs.Conj(ez)

        Sx_TE = -1j/(2*self._ngsolve_omega*const_mu_0)*0.5*(
            ez*ez.Deriv()/ngs.Conj(self._cf_sx*self._cf_mu_raw)
            -ngs.Conj(ez*ez.Deriv())/(self._cf_sx*self._cf_mu_raw)
            )

        # Get the indices of points in three different regions
        ipx_wg = np.where( (self.mesh.px>=self.geom.region_wg[0]) & (self.mesh.px<=self.geom.region_wg[1]) )
        ipx_phy = np.where( (self.mesh.px>=self.geom.region_phy[0]) & (self.mesh.px<=self.geom.region_phy[1]) )

        # px_pml_lhalf = [self.geom.intervals[0], 0.5*(self.geom.intervals[0]+self.geom.intervals[1])]
        # ipx_pml_lhalf = np.where( (self.mesh.px>=px_pml_lhalf[0]) & (self.mesh.px<=px_pml_lhalf[1]) )

        # px_pml_rhalf = [0.5*(self.geom.intervals[-2]+self.geom.intervals[-1]), self.geom.intervals[-1]]
        # ipx_pml_rhalf = np.where( (self.mesh.px>=px_pml_rhalf[0]) & (self.mesh.px<=px_pml_rhalf[1]) )

        #### conditions for mode sorting
        # and filtering by several conditions:
        # 1. "<_epsilon_max[1]*1.1" rather than "<_epsilon_max[1]*1.0" to
        # be robust against possible numerical inaccuracy when a mode's
        # effective epsilon is slightly higher than the largest material epsilon
        #
        # 2. Only attenuating mode, but "neff_TM[i].imag>-1e-2" still
        # allows a small negative imaginary part not exceeding "-1e-2"
        # for robustness (possible numerical inaccuracy of near-zero-loss cases)
        #
        # 3. Confinement factor
        #
        # 4. Quality factor
        #
        # For better performance,
        # the conditions 1, 2, 4 may be applied before the postprocessing in the for loop
        idx_TM = np.where(
            (np.abs(self.sol.epsilon_TM.real)<self._epsilon_max[1]*1.1)
            & (self.sol.neff_TM.imag>-1e-2)
            & (np.abs(self.sol.neff_TM.imag/self.sol.neff_TM.real)<10)
        )[0]

        idx_TE = np.where(
            (np.abs(self.sol.epsilon_TE.real)<self._epsilon_max[1]*1.1)
            & (self.sol.neff_TE.imag>-1e-2)
            & (np.abs(self.sol.neff_TE.imag/self.sol.neff_TE.real)<10)
        )[0]

        #### TM mode sorting
        for i in idx_TM:
            beta_re.Set(self.sol.neff_TM[i].real*self.k0)
            beta_im.Set(self.sol.neff_TM[i].imag*self.k0)

            # working on the ith mode profile
            hz.vec.data = self.sol.hz.vecs[i].data

            # Divided by the maximum, normalize to a unity amplitude
            hz_vec = hz.vec.data[:self.mesh.npoints].FV().NumPy()
            imax_TM = np.nanargmax(np.abs(hz_vec))
            sign_peak = 1

            # # always force the leftmost peak in the waveguide region to be positive
            # hz_seg = hz_vec[ipx_wg]
            # # hz_seg = hz_vec[ipx_phy]
            # ipeaks, _ = scipy_signal.find_peaks(np.abs(hz_seg), distance=5)
            # if ipeaks.size>0 and hz_seg[ipeaks[0]].real<0:
            #     sign_peak = -1

            hz.vec.data *= sign_peak/hz_vec[imax_TM]

            # Normalize to the total power
            # if normalization:
            #     int_Sy_TM = ngs.Integrate(Sy_TM, self.mesh._ngsolve_mesh, order=4)
            #     hz.vec.data *= 1/(int_Sy_TM)**0.5
            # else:
            #     int_Sy_TM = 1

            # evaluation on mesh points
            for ip, p in zip(range(self.mesh.npoints), self.mesh._ngsolve_mesh.ngmesh.Points()):
                self.sol.Sy_TM[i, ip] = Sy_TM(self.mesh._ngsolve_mesh(p[0]))
                # self.sol.Sx_TM[i, ip] = Sx_TM(self.mesh._ngsolve_mesh(p[0]))

                self.sol.ue_TM[i, ip] = ue_TM(self.mesh._ngsolve_mesh(p[0]))
                self.sol.um_TM[i, ip] = um_TM(self.mesh._ngsolve_mesh(p[0]))

                # self.sol.Qe_TM[i, ip] = Qe_TM(self.mesh._ngsolve_mesh(p[0]))
                # self.sol.Qm_TM[i, ip] = Qm_TM(self.mesh._ngsolve_mesh(p[0]))

                self.sol.hz_TM[i, ip] = hz(self.mesh._ngsolve_mesh(p[0]))
                self.sol.ex_TM[i, ip] = ex(self.mesh._ngsolve_mesh(p[0]))
                self.sol.ey_TM[i, ip] = ey(self.mesh._ngsolve_mesh(p[0]))

            int_phy_TM = ngs.Integrate( stepfun_phy*(u_TM), self.mesh._ngsolve_mesh, order=4)
            int_pml_left_TM = ngs.Integrate( stepfun_pml_left*(u_TM), self.mesh._ngsolve_mesh, order=4)
            int_pml_right_TM = ngs.Integrate( stepfun_pml_right*(u_TM), self.mesh._ngsolve_mesh, order=4)
            int_tot_TM = ngs.Integrate( u_TM, self.mesh._ngsolve_mesh, order=4)
            # self.sol.kappa_TM[i] = np.abs(int_phy_TM.real/int_tot_TM.real)
            self.sol.kappa_left_TM[i] = np.abs(int_phy_TM.real/int_pml_left_TM.real)
            self.sol.kappa_right_TM[i] = np.abs(int_phy_TM.real/int_pml_right_TM.real)
            int_wg_TM = ngs.Integrate( stepfun_wg*(u_TM), self.mesh._ngsolve_mesh, order=4)
            # self.sol.kappa_TM[i] = np.abs(int_wg_TM.real/int_tot_TM.real)

            # power flow
            int_Sy_phy_TM = ngs.Integrate( stepfun_phy*(Sy_TM), self.mesh._ngsolve_mesh, order=4)
            int_Sy_pml_left_TM = ngs.Integrate( stepfun_pml_left*(Sy_TM), self.mesh._ngsolve_mesh, order=4)
            int_Sy_pml_right_TM = ngs.Integrate( stepfun_pml_right*(Sy_TM), self.mesh._ngsolve_mesh, order=4)
            self.sol.kappa_Syleft_TM[i] = np.abs(int_Sy_phy_TM.real/int_Sy_pml_left_TM.real)
            self.sol.kappa_Syright_TM[i] = np.abs(int_Sy_phy_TM.real/int_Sy_pml_right_TM.real)

            # wdiff_wg_TM = ngs.Integrate( stepfun_wg*(ue_TM-um_TM), self.mesh._ngsolve_mesh, order=4)
            # wdiff_wg_TM = ngs.Integrate( stepfun_wg*(udiff2_TM), self.mesh._ngsolve_mesh, order=4)
            # int_u2_wg_TM = ngs.Integrate( stepfun_wg*(u_TM**2), self.mesh._ngsolve_mesh, order=4)
            max_udiff2_wg_TM = np.nanmax( np.abs(self.sol.ue_TM[i, ipx_wg] - self.sol.um_TM[i, ipx_wg]) )**2
            max_udiff2_TM = np.nanmax( np.abs(self.sol.ue_TM[i, :] - self.sol.um_TM[i, :]) )**2
            self.sol.kappa_wdiff_TM[i] = np.abs(max_udiff2_wg_TM/max_udiff2_TM)
            # self.sol.kappa_wdiff_TM[i] = np.abs(wdiff_wg_TM/max_udiff2_TM)
            # wdiff_tot_TM = ngs.Integrate( stepfun_phy*(udiff2_TM), self.mesh._ngsolve_mesh, order=4)
            # self.sol.kappa_wdiff_TM[i] = np.abs(wdiff_wg_TM/wdiff_tot_TM)

            # the ratio of two maxima
            # max_wg_TM = ngs.Integrate( stepfun_wg*(u_TM), self.mesh._ngsolve_mesh, order=4)
            # max_wg_TM = np.nanmax( np.abs(self.sol.ue_TM[i, ipx_wg] - self.sol.um_TM[i, ipx_wg]) )
            # max_pml_lhalf_TM = np.nanmax( np.abs(self.sol.ue_TM[i, ipx_pml_lhalf]
            #                                       - self.sol.um_TM[i, ipx_pml_lhalf]) )
            # max_pml_rhalf_TM = np.nanmax( np.abs(self.sol.ue_TM[i, ipx_pml_rhalf]
            #                                       - self.sol.um_TM[i, ipx_pml_rhalf]) )
            # max_pml_lhalf_TM = np.nanmax( np.abs(self.sol.ue_TM[i, 0]
            #                                       - self.sol.um_TM[i, 0]) )
            # max_pml_rhalf_TM = np.nanmax( np.abs(self.sol.ue_TM[i, self.mesh.npoints-1]
            #                                       - self.sol.um_TM[i, self.mesh.npoints-1]) )

            # kappa_lhalf_TM = max_wg_TM/max_pml_lhalf_TM
            # kappa_rhalf_TM = max_wg_TM/max_pml_rhalf_TM
            # self.sol.kappa_TM[i] = np.nanmin([kappa_lhalf_TM, kappa_rhalf_TM])
            # print(max_wg_TM, max_pml_lhalf_TM, max_pml_rhalf_TM, kappa_lhalf_TM, kappa_rhalf_TM); continue

            # filtering by mode confinement
            if (self.sol.kappa_left_TM[i]>self.sol.kappa_threshold
                and self.sol.kappa_right_TM[i]>self.sol.kappa_threshold
                and self.sol.kappa_Syleft_TM[i]>self.sol.kappa_threshold
                and self.sol.kappa_Syright_TM[i]>self.sol.kappa_threshold
                ):
                self.sol.filtered_TM[i] = True

        #### TE mode sorting
        for i in idx_TE:
        # for i, neff in enumerate(self.sol.neff_TE):
            beta_re.Set(self.sol.neff_TE[i].real*self.k0)
            beta_im.Set(self.sol.neff_TE[i].imag*self.k0)

            # working on the ith mode profile
            ez.vec.data = self.sol.ez.vecs[i].data

            # Divided by the maximum, normalize to a unity amplitude
            ez_vec = ez.vec.data[:self.mesh.npoints].FV().NumPy()
            imax_TE = np.nanargmax(np.abs(ez_vec))
            ez.vec.data *= 1/ez_vec[imax_TE]

            # Normalize to the total power
            # if normalization:
            #     int_Sy_TE = ngs.Integrate( Sy_TE, self.mesh._ngsolve_mesh, order=4)
            #     ez.vec.data *= 1/(int_Sy_TE)**0.5
            # else:
            #     int_Sy_TE = 1

            # evaluation on mesh points
            for ip, p in zip(range(self.mesh.npoints), self.mesh._ngsolve_mesh.ngmesh.Points()):
                self.sol.Sy_TE[i, ip] = Sy_TE(self.mesh._ngsolve_mesh(p[0]))
                self.sol.Sx_TE[i, ip] = Sx_TE(self.mesh._ngsolve_mesh(p[0]))

                self.sol.ue_TE[i, ip] = ue_TE(self.mesh._ngsolve_mesh(p[0]))
                self.sol.um_TE[i, ip] = um_TE(self.mesh._ngsolve_mesh(p[0]))

                self.sol.Qe_TE[i, ip] = Qe_TE(self.mesh._ngsolve_mesh(p[0]))
                self.sol.Qm_TE[i, ip] = Qm_TE(self.mesh._ngsolve_mesh(p[0]))

                self.sol.ez_TE[i, ip] = ez(self.mesh._ngsolve_mesh(p[0]))
                self.sol.hx_TE[i, ip] = hx(self.mesh._ngsolve_mesh(p[0]))
                self.sol.hy_TE[i, ip] = hy(self.mesh._ngsolve_mesh(p[0]))

            int_wg_TE = ngs.Integrate( stepfun_wg*(u_TE), self.mesh._ngsolve_mesh, order=4)
            int_phy_TE = ngs.Integrate( stepfun_phy*(u_TE), self.mesh._ngsolve_mesh, order=4)
            int_pml_left_TE = ngs.Integrate( stepfun_pml_left*(u_TE), self.mesh._ngsolve_mesh, order=4)
            int_pml_right_TE = ngs.Integrate( stepfun_pml_right*(u_TE), self.mesh._ngsolve_mesh, order=4)
            # int_tot_TE = ngs.Integrate( u_TE, self.mesh._ngsolve_mesh, order=4)
            # self.sol.kappa_TE[i] = np.abs(int_wg_TE.real/int_tot_TE.real)
            self.sol.kappa_left_TE[i] = np.abs(int_phy_TE.real/int_pml_left_TE.real)
            self.sol.kappa_right_TE[i] = np.abs(int_phy_TE.real/int_pml_right_TE.real)

            # power flow
            int_Sy_phy_TE = ngs.Integrate( stepfun_phy*(Sy_TE), self.mesh._ngsolve_mesh, order=4)
            int_Sy_pml_left_TE = ngs.Integrate( stepfun_pml_left*(Sy_TE), self.mesh._ngsolve_mesh, order=4)
            int_Sy_pml_right_TE = ngs.Integrate( stepfun_pml_right*(Sy_TE), self.mesh._ngsolve_mesh, order=4)
            self.sol.kappa_Syleft_TE[i] = np.abs(int_Sy_phy_TE.real/int_Sy_pml_left_TE.real)
            self.sol.kappa_Syright_TE[i] = np.abs(int_Sy_phy_TE.real/int_Sy_pml_right_TE.real)

            # wdiff_wg_TE = ngs.Integrate( stepfun_wg*(ue_TE-um_TE), self.mesh._ngsolve_mesh, order=4)
            # wdiff_wg_TE = ngs.Integrate( stepfun_wg*(udiff2_TE), self.mesh._ngsolve_mesh, order=4)
            # int_u2_wg_TE = ngs.Integrate( stepfun_wg*(u_TE**2), self.mesh._ngsolve_mesh, order=4)
            # wdiff_tot_TE = ngs.Integrate( stepfun_phy*(udiff2_TE), self.mesh._ngsolve_mesh, order=4)
            max_udiff2_wg_TE = np.nanmax( np.abs(self.sol.ue_TE[i, ipx_wg] - self.sol.um_TE[i, ipx_wg]) )**2
            max_udiff2_TE = np.nanmax( np.abs(self.sol.ue_TE[i, :] - self.sol.um_TE[i, :]) )**2
            # self.sol.kappa_wdiff_TE[i] = np.abs(wdiff_wg_TE/wdiff_tot_TE)
            self.sol.kappa_wdiff_TE[i] = np.abs(max_udiff2_wg_TE/max_udiff2_TE)
            # self.sol.kappa_wdiff_TE[i] = np.abs(wdiff_wg_TE/max_udiff2_TE)
            # 2) the ratio of two maxima
            # max_wg_TE = ngs.Integrate( stepfun_wg*(u_TE), self.mesh._ngsolve_mesh, order=4)
            # max_wg_TE = np.nanmax( np.abs(self.sol.ue_TE[i, ipx_wg] - self.sol.um_TE[i, ipx_wg]) )
            # max_pml_lhalf_TE = np.nanmax( np.abs(self.sol.ue_TE[i, ipx_pml_lhalf]
            #                                       - self.sol.um_TE[i, ipx_pml_lhalf]) )
            # max_pml_rhalf_TE = np.nanmax( np.abs(self.sol.ue_TE[i, ipx_pml_rhalf]
            #                                       - self.sol.um_TE[i, ipx_pml_rhalf]) )
            # max_pml_lhalf_TE = np.nanmax( np.abs(self.sol.ue_TE[i, 0]
            #                                       - self.sol.um_TE[i, 0]) )
            # max_pml_rhalf_TE = np.nanmax( np.abs(self.sol.ue_TE[i, self.mesh.npoints-1]
            #                                       - self.sol.um_TE[i, self.mesh.npoints-1]) )

            # kappa_lhalf_TE = max_wg_TE/max_pml_lhalf_TE
            # kappa_rhalf_TE = max_wg_TE/max_pml_rhalf_TE
            # self.sol.kappa_TE[i] = np.nanmin([kappa_lhalf_TE, kappa_rhalf_TE])

            # filtering
            if(self.sol.kappa_left_TE[i]>self.sol.kappa_threshold
               and self.sol.kappa_left_TE[i]>self.sol.kappa_threshold
               and self.sol.kappa_Syleft_TE[i]>self.sol.kappa_threshold
               and self.sol.kappa_Syleft_TE[i]>self.sol.kappa_threshold
               ):
                self.sol.filtered_TE[i] = True

    def Solve(self, method="ngsolve_ArnoldiSolver", keepInMem=False, show_pattern=False):

        print("Solving at the normalized frequency w = "+"{0.real:.3f}".format(self.w))

        #%% auxiliary variables, for post-processing and filtering

        # get values of epsilon and mu at the current frequency
        self._mu_raw = np.empty((self.mesh.npoints), dtype=np.complex64)
        self._mu = np.empty((self.mesh.npoints, 3), dtype=np.complex64) # 3 digonal elements in a tensor
        self._epsilon_raw = np.empty((self.mesh.npoints), dtype=np.complex64)
        self._epsilon = np.empty((self.mesh.npoints, 3), dtype=np.complex64) # 3 digonal elements in a tensor
        self._sx = np.empty((self.mesh.npoints), dtype=np.complex64)

        # For investigating the PML scaling factor sx and epsilon/mu, before and after enabling PML
        for i, p in zip(range(self.mesh.npoints), self.mesh._ngsolve_mesh.ngmesh.Points()):
            self._sx[i] = self._cf_sx(self.mesh._ngsolve_mesh(p[0]))

            self._mu_raw[i] = self._cf_mu_raw(self.mesh._ngsolve_mesh(p[0]))
            self._epsilon_raw[i] = self._cf_epsilon_raw(self.mesh._ngsolve_mesh(p[0]))

            self._mu[i,:] = self._mu_raw[i]*np.array([1/self._sx[i], self._sx[i], self._sx[i]])
            self._epsilon[i,:] = self._epsilon_raw[i]*np.array([1/self._sx[i], self._sx[i], self._sx[i]])

        # print('epsilon: ', self._epsilon)
        # print('mu: ', self._mu)
        # print('epsilon_raw: ', self._epsilon_raw)
        # print('mu_raw: ', self._mu_raw)
        # sys.exit(0)

        # list of complex refractive indices segment by segment
        n2list = []
        intervals = np.asarray(self.geom.intervals)
        mid_intervals = 0.5*(intervals[:-1]+intervals[1:])
        for p in mid_intervals:

            mu_raw = self._cf_mu_raw(self.mesh._ngsolve_mesh(p))
            epsilon_raw = self._cf_epsilon_raw(self.mesh._ngsolve_mesh(p))

            if np.abs(mu_raw-1)>1e-4:
                print("Be careful interpreting the refractive index when the permeability is not 1!")
            n2list.append(mu_raw*epsilon_raw)

        self.material.nklist = np.asarray(n2list)**0.5


        #%% solve with TBC
        if self.TBC:

            self._k2left = self.k0**2*self._epsilon_raw[0]*self._mu_raw[0]
            self._k2right = self.k0**2*self._epsilon_raw[-1]*self._mu_raw[-1]
            self._n2left = self._epsilon_raw[0]*self._mu_raw[0]
            self._n2right = self._epsilon_raw[-1]*self._mu_raw[-1]

            n2max = np.max(np.abs(n2list))
            nmax = n2max**0.5
            kmax = nmax*self.k0
            k2max = n2max*self.k0
            # print(nmax, self.k0); sys.exit(0);

            n2_sum = (self._n2right + self._n2left)
            n2_diff = (self._n2right - self._n2left)
            Sigma2 = n2_sum
            delta2 = n2_diff

            # convert to PETSc matrices
            def ngs2petscMatAIJ(ngs_mat):
                locmat = ngs_mat.local_mat
                eh, ew = locmat.entrysizes
                val,col,ind = locmat.CSR()
                ind = np.array(ind).astype(PETSc.IntType)
                col = np.array(col).astype(PETSc.IntType)
                apsc_loc = PETSc.Mat().createBAIJ(size=(eh*locmat.height, eh*locmat.width), bsize=eh, csr=(ind,col,val))

                return apsc_loc

            # directly convert from ngsolve matrix to numpy matrix
            def ngs2numpyMatList(ngs_matList):
                numpy_matList = []
                for ngs_mat in ngs_matList:
                    rows,cols,vals = ngs_mat.mat.COO()
                    numpy_matList.append(scipy_sparse.csr_matrix((vals,(rows,cols))).todense())

                return numpy_matList

            def ngs2petscMatAIJList(ngs_matList):
                petsc_matList = []
                for ngs_mat in ngs_matList:
                    #!!! the matrix conversion provided by ngs is wrong,
                    # as the two outermost dimensions are removed.
                    # petsc_matList.append(ngs.ngs2petsc.CreatePETScMatrix(ngs_mat.mat, self.fes.FreeDofs()))

                    petsc_matList.append(ngs2petscMatAIJ(ngs_mat.mat))

                return petsc_matList

            def assembleList(Alist):
                for Ai in Alist:
                    Ai.Assemble()
                pass

            # investigate elements of the assembled matrices
            def petscMat2numpyMat(v):
                s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
                return s

            assembleList(self._AlistTM)
            assembleList(self._AlistTE)

            # self._testTM.Assemble()
            # rows,cols,vals = self._testTM.mat.COO()
            # print(rows, cols, vals); sys.exit(0)

            petsc_AListTM = ngs2petscMatAIJList(self._AlistTM)
            petsc_AListTE = ngs2petscMatAIJList(self._AlistTE)

            # # load data calculated from `nlevp` package, where the linear finite element is used.
            # # [coeffs,fun,F,xcoeffs] = planar_waveguide; where planar_waveguide is given
            # # in the package nlevp
            # nlevp_4layerwg = scipy_io.loadmat("4layer_wg.mat")
            # nlevp_4layerwg_coeffs = nlevp_4layerwg["coeffs"]
            # # print(np.shape(nlevp_4layerwg_coeffs[0, 0]))
            # # print(self.fes.ndof)#, petsc_AListTM[0].shape)
            # # sys.exit(0)

            # petsc_AListTM = ngs2petscMatAIJList(self._AlistTE)
            # petsc_AListTE = [
            #     PETSc.Mat().createDense(nlevp_4layerwg_coeffs[0,0].shape, array=nlevp_4layerwg_coeffs[0,0]),
            #     PETSc.Mat().createDense(nlevp_4layerwg_coeffs[0,1].shape, array=nlevp_4layerwg_coeffs[0,1]),
            #     PETSc.Mat().createDense(nlevp_4layerwg_coeffs[0,2].shape, array=nlevp_4layerwg_coeffs[0,2]),
            #     PETSc.Mat().createDense(nlevp_4layerwg_coeffs[0,3].shape, array=nlevp_4layerwg_coeffs[0,3]),
            #     PETSc.Mat().createDense(nlevp_4layerwg_coeffs[0,4].shape, array=nlevp_4layerwg_coeffs[0,4]),
            #     ]
            # # sys.exit(0)

            #### investigate the system matrices
            # print(self._petsc_A0TM.getSize())
            investigate = True;
            investigate = False;

            if investigate:
                fig, axs = _plt.subplots(figsize = (6*_fig.h*_fig.inch2cm*_fig.zoom, 2.8*_fig.h*_fig.inch2cm*_fig.zoom), ncols = 5, nrows = 4, constrained_layout = True);

                for ic in range(5):
                    numpy_ATM = petscMat2numpyMat(petsc_AListTM[ic])
                    numpy_ATE = petscMat2numpyMat(petsc_AListTE[ic])
                    # print(np.divide(numpy_ATM, numpy_ATE))
                    acb = axs[0, ic].imshow(np.real(numpy_ATM)); _plt.colorbar(acb, ax=axs[0,ic]);
                    acb = axs[1, ic].imshow(np.imag(numpy_ATM)); _plt.colorbar(acb, ax=axs[1,ic]);
                    acb = axs[2, ic].imshow(np.real(numpy_ATE)); _plt.colorbar(acb, ax=axs[2,ic]);
                    acb = axs[3, ic].imshow(np.imag(numpy_ATE)); _plt.colorbar(acb, ax=axs[3,ic]);
                sys.exit(0)
            # print(dTM[:self.fes.ndof*2+4,:self.fes.ndof*2+4]); sys.exit(0)

            # asymmetric cover and substrate layers
            if np.abs(n2_diff)>1e-12:
                pass
            # symmetric cover and substrate layers
            # \delta^2 is equal to zero, resulting in frist two lowest order matrices being trivial
            else:
                petsc_AListTM = [petsc_AListTM[2], petsc_AListTM[3], petsc_AListTM[4]]
                petsc_AListTE = [petsc_AListTE[2], petsc_AListTE[3], petsc_AListTE[4]]

            method_slepc = True
            # method_slepc = False

            # Setup the number of eigensolvers to be sought
            # neigs_query = min(self.fes.ndof, 200) # the number of eigenvalues is potentially self.fes.ndof*4
            neigs_query = self.fes.ndof*4
            # neigs_query = self.fes.ndof # the number of eigenvalues is potentially self.fes.ndof*4

            # we use the nonlinear FEAST algorithm described in the paper:
            # [1] B. Gavin, A. Międlar, and E. Polizzi, ‘FEAST eigensolver for nonlinear eigenvalue problems’, Journal of Computational Science, vol. 27, pp. 107–117, Jul. 2018, doi: 10.1016/j.jocs.2018.05.006.

            #### customized NLFEAST
            # if method_slepc==False:

            #     # contour integral in the nonlinear FEAST method
            #     # the radius for the contour integral
            #     # z_rad = self.k0 # if lambda = (k_cx + k_sx)/2
            #     z_rad = 1 # if lambda = (tau_c + tau_s)/2
            #     z_0 = 0+0j
            #     z_num = 11

            #     # parameterized contour z(t)
            #     def contour(t):
            #         return z_0 + z_rad*np.cos(t) + 1j*z_rad*np.sin(t)

            #     # dz/dt
            #     def jacobian(t):
            #         return -z_rad*np.sin(t) + 1j*z_rad*np.cos(t)

            #     # each term is A_i*z^i
            #     def AList2TzList(AList, z):
            #         TzList = AList
            #         for i in range(len(AList)):
            #             TzList[i] = AList[i]*z**i
            #         return TzList

            #     def TzList2TzSum(TzList):
            #         Tz = TzList[0]
            #         for i in range(1, len(TzList)):
            #             Tz += TzList[i]

            #         return Tz

            #     t_points = np.delete(np.linspace(0, 2*np.pi, z_num+1) + 2*np.pi/(2*z_num), -1)
            #     quad_points = contour(t_points)
            #     quad_jacobians = contour(t_points)
            #     quad_weights = (2*np.pi)/z_num * np.ones((z_num))
            #     # print(quad_points)
            #     # _plt.scatter(quad_points.real, quad_points.imag, c=range(z_num))
            #     # sys.exit(0)

            #     # numpy random number generation to generate nxm matrix, where
            #     # n is the the dimension of any matrix a to e, and
            #     # m is the desired number of eigenvalues searched in a given contour
            #     # in the complex plane
            #     # x = np.random.uniform(-1, 1, self.fes.ndof)
            #     X = np.empty((self.fes.ndof, neigs_query), dtype=np.complex128)
            #     for m in range(neigs_query):
            #         X[:, m] = np.sqrt(np.random.uniform(0, 1, self.fes.ndof)) * np.exp(1j * np.random.uniform(0, 2 * np.pi, self.fes.ndof))
            #         # _plt.plot(X[:, m].real, X[:, m].imag, 'o')

            #     #FIXME: try QR matrix factorization from PETSc instead of from numpy.linalg.qr()
            #     Q, R = np.linalg.qr(np.matrix(X))
            #     petsc_Q = PETSc.Mat().createDense(Q.shape, array=Q)

            #     # reduced nonlinear eigenvalue problem, Eq. (15) ref [1]
            #     rNEP = SLEPc.PEP().create()
            #     rNEP.setTolerances(1.0e-12, 200)
            #     rNEP.setDimensions(neigs_query) # number of requested eigenvalues
            #     rNEP.setProblemType(SLEPc.PEP.ProblemType.GENERAL)

            #     err = 1; tol = 1e-4; iter = 0; iter_max=2
            #     while err > tol and iter<iter_max:
            #         print("iter:", iter)

            #         petsc_Q.assemble()

            #         # convert to PETSc matrix
            #         # petsc_QH = PETSc.Mat().createDense(Q.shape, array=Q)
            #         petsc_QH = petsc_Q.duplicate(copy=True)
            #         petsc_QH.hermitianTranspose() # in-place hermitian transpose

            #         petsc_aListTM = []
            #         for i in range(len(petsc_AListTM)):
            #             petsc_aListTM.append(petsc_QH.matMatMult(petsc_AListTM[i], petsc_Q))

            #         # update R values
            #         rNEP.setOperators(petsc_aListTM)
            #         rNEP.setFromOptions()
            #         rNEP.solve() # a nonlinear (quartic) eigenvalue problem, potentially generate 4 times eigenvalues

            #         # create the vector x to be assigned to Q*y
            #         petsc_x = petsc_Q.createVecs(side='left')

            #         # where y is to be solved as the eigen vector of the reduced problem
            #         petsc_y = petsc_Q.createVecs(side='right')

            #         # create a list of the residual vectors rList to be assigned to
            #         # lambda_i^p*A_p*x = lambda_i^p*A_p*Q*y
            #         petsc_rList = []
            #         for i in range(len(petsc_AListTM)):
            #             petsc_rList.append(petsc_AListTM[i].createVecs(side='right'))

            #         nconvergedTM = rNEP.getConverged()
            #         eigvals_all = np.empty((max(nconvergedTM, neigs_query)), dtype=np.complex128)
            #         residuals = np.empty((max(nconvergedTM, neigs_query)), dtype=np.float64)

            #         # pertsc_r: residual function associated with a single eigenvalue
            #         petsc_r = PETSc.Mat().createDense((self.fes.ndof))

            #         # pertsc_R: block form of the residual function associated with all eigenvalues
            #         petsc_R = PETSc.Mat().createDense((self.fes.ndof, max(nconvergedTM, neigs_query)))

            #         # petsc_X: block form of the newly approximated eigen vectors
            #         petsc_X = PETSc.Mat().createDense((self.fes.ndof, max(nconvergedTM, neigs_query)))

            #         # shift-and-invert of the eigenvalues diagonal matrix
            #         # (zI-\Lambda)^(-1)
            #         petsc_eigvalsShiftInv = PETSc.Mat().create()
            #         petsc_eigvalsShiftInv.setFromOptions()
            #         petsc_eigvalsShiftInv.setSizes(nconvergedTM)

            #         for i in range(nconvergedTM):

            #             # FIXME: strange behavior in slepc4py that
            #             # actually the 1st vector yr gets both the real and imaginary parts
            #             # whereas the 2nd vector yi gets nothing
            #             eigvals_all[i] = rNEP.getEigenpair(i, petsc_y)

            #             # x = Qy
            #             # print(petsc_y.getValues(range(neigs_query)))
            #             petsc_Q.mult(petsc_y, petsc_x)
            #             # print(petsc_x.getValues(range(self.fes.ndof)))

            #             # update the eigen vector associated with a single eigenvalue
            #             petsc_X.setValues(range(self.fes.ndof), i, petsc_x.getValues(range(self.fes.ndof)))

            #             # T(z) at z = eigvals_all[i]
            #             petsc_TlambdaList = AList2TzList(petsc_AListTM, eigvals_all[i])

            #             # T(z)*x
            #             for k in range(len(petsc_TlambdaList)):
            #                 petsc_TlambdaList[k].mult(petsc_x, petsc_rList[k])
            #                 # print(petsc_rList[k].view())

            #             # Eq. (14), the sum vector \Sigma_p A_p*x*lambda_i^p
            #             petsc_r = petsc_rList[0] +  petsc_rList[1] + petsc_rList[2] + petsc_rList[3] + petsc_rList[4]

            #             # update the ith column in T(X, \Lambda)
            #             petsc_R.setValues(range(self.fes.ndof), i, petsc_r.getValues(range(self.fes.ndof)))

            #             # the norm: ||T(lambda_i)*Q*y_i|| in Eq. (14) or in Step 3 below Eq. (16)
            #             residuals[i] = petsc_r.norm()

            #         # keep eigen values inside the contour
            #         # eigvals_all = np.where(np.abs(eigvals_all-z_0)<z_rad, eigvals_all, np.nan+1j*np.nan)
            #         # sort out the smallest neigs_query eigen values
            #         idx = np.argsort(np.abs(eigvals_all))
            #         # print(residuals)
            #         err = np.nanargmax(residuals[idx[:neigs_query]])

            #         # update the Q matrix by contour integral
            #         petsc_R.assemble()
            #         petsc_X.assemble()
            #         # print(petsc_R.getSize(), neigs_query)

            #         # projection
            #         petsc_P = petsc_R.duplicate(copy=False)
            #         petsc_sumQ = petsc_R.duplicate(copy=False)

            #         # Calculate a part of the integrand in Eq. (13), and
            #         # that part is petsc_R = T(X, \Lambda) left multiplied by T^(-1).
            #         # The matrix inversion is avoided by solving a linear equation of the form A*x = b
            #         ksp = PETSc.KSP()
            #         ksp.create()
            #         for iq in range(z_num):

            #             petsc_Tz = TzList2TzSum(AList2TzList(petsc_AListTM, quad_points[iq]))
            #             ksp.setOperators(petsc_Tz)
            #             ksp.setFromOptions()
            #             ksp.matSolve(petsc_R, petsc_P)
            #             petsc_Pshift = petsc_X - petsc_P

            #             # update the shift-and-invert value associated with the quadrature point
            #             for i in range(nconvergedTM):
            #                 petsc_eigvalsShiftInv.setValue(i, i, 1/(quad_points[iq]-eigvals_all[i]))

            #             petsc_eigvalsShiftInv.assemble()

            #             petsc_quadQ = petsc_Pshift.matMult(petsc_eigvalsShiftInv)

            #             petsc_sumQ += petsc_quadQ*quad_jacobians[iq]*quad_weights[iq]

            #         print("error:", err)

            #         # sys.exit(0)
            #         iter += 1
            #         petsc_sumQ.assemble()
            #         # petsc_Q.setValues(range(self.fes.ndof), range(neigs_query), petsc_sumQ.getValues(range(self.fes.ndof), range(neigs_query)))

            #         # QR factorization by numpy
            #         new_Q, new_R = np.linalg.qr(petscMat2numpyMat(petsc_Q))
            #         petsc_Q = PETSc.Mat().createDense(new_Q.shape, array=new_Q)
            #         print(petsc_Q.view())

            #         sys.exit(0)

            #         # FIXME: a direct QR factorization using PETSc or SLEPc seems impossible.
            #         # test qr factorization by PETSc
            #         # numpy_qr = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
            #         # test_q, test_r = np.linalg.qr(numpy_qr)
            #         # # print(test_q)
            #         # test_qr = PETSc.Mat().createDense(numpy_qr.shape, array=numpy_qr)
            #         # test_bv = SLEPc.BV().create()
            #         # # test_bv.setMatrix(test_qr, True)
            #         # test_bv.orthogonalize(test_qr)
            #         # test_q, test_r = test_bv.getMatrix()
            #         # print(test_q.view())

            #         # test_b = PETSc.Vec().createWithArray(np.array([1, 1, 1]))
            #         # test_x = PETSc.Vec().createWithArray(np.array([1, 1, 1]))
            #         # test_ksp = PETSc.KSP()
            #         # test_ksp.create()
            #         # test_ksp.getPC().setType('qr')
            #         # pc = test_ksp.getPC()
            #         # test_ksp.setOperators(test_qr)
            #         # test_ksp.setFromOptions()
            #         # test_ksp.solve(test_b, test_x)
            #         # a, m = pc.getOperators()
            #         # print(a.view())
            #         # print(m.view())

            #     sys.exit(0)

            #### SLEPc.PEP()
            if method_slepc:
            # else:
                if self._slepc_PEP:
                    self._QTM = SLEPc.PEP().create()
                    self._QTM.setOperators(petsc_AListTM)
                    self._QTM.setTolerances(1.0e-8, 50)
                    self._QTM.setDimensions(neigs_query) # number of requested eigenvalues
                    self._QTM.setProblemType(SLEPc.PEP.ProblemType.GENERAL)
                    # setting to other problem type do not work well.

                    # eigenvalues in a contour
                    #
                    # the contour is determined by vertices of a polygon
                    #
                    # 5xnmax is enough for numerical simulations, even in the case of SPP modes.

                    a = nmax*100.1; b = nmax*100.1;
                    z_rad = 1 # if lambda = (tau_c + tau_s)/2
                    # z_rad = self.k0 # if lambda = (k_cx + k_sx)/2
                    vertices_1 = np.array([z_rad*0.1-1j*z_rad*0.1,
                                          z_rad*a-1j*z_rad*b,
                                          z_rad*a*np.sqrt(2)+1j*z_rad*0,
                                          z_rad*a+1j*z_rad*b,
                                          -z_rad*0.1+1j*z_rad*b,
                                          -z_rad*0+1j*z_rad*0.1,
                                          ])
                    vertices_2 = np.array([-z_rad*a-1j*z_rad*b,
                                          z_rad*a-1j*z_rad*b,
                                          z_rad*a+1j*z_rad*b,
                                          -z_rad*a+1j*z_rad*b,
                                          ])
                    # vertices = vertices_1
                    vertices = vertices_2

                    slepc_rg = self._QTM.getRG()
                    slepc_rg.setType(SLEPc.RG.Type.POLYGON)
                    slepc_rg.setPolygonVertices(vertices)

                    # in the polynomial nonlinear eigenvalue formulation
                    # The default monomial basis works the best
                    self._QTM.setBasis(SLEPc.PEP.Basis.MONOMIAL) # default Basis
                    # the other bases do not work in our case.

                    self._QTM.setFromOptions()
                    # Solve the eigensystem
                    self._QTM.solve()

                    self._QTE = SLEPc.PEP().create()
                    self._QTE.setOperators(petsc_AListTE)
                    self._QTE.setTolerances(1.0e-8, 50)
                    self._QTE.setDimensions(neigs_query) # number of requested eigenvalues
                    self._QTE.setProblemType(SLEPc.PEP.ProblemType.GENERAL)

                    # eigenvalues in a contour
                    slepc_rg = self._QTE.getRG()
                    slepc_rg.setType(SLEPc.RG.Type.POLYGON)
                    slepc_rg.setPolygonVertices(vertices)

                    self._QTE.setBasis(SLEPc.PEP.Basis.MONOMIAL)

                    self._QTE.setFromOptions()
                    # Solve the eigensystem
                    self._QTE.solve()

                    petsc_yTM = petsc_AListTM[0].createVecs(side='right')
                    petsc_yTE = petsc_AListTE[0].createVecs(side='right')

                    nconvergedTM = self._QTM.getConverged()
                    eigvals_TM = np.empty((nconvergedTM), dtype=np.complex128)
                    eigvals_TM[:] = np.nan+1j*np.nan

                    nconvergedTE = self._QTE.getConverged()
                    eigvals_TE = np.empty((nconvergedTE), dtype=np.complex128)
                    eigvals_TE[:] = np.nan+1j*np.nan

                    # For saving solution eigen vectors
                    self.sol.hz = ngs.GridFunction(self.fes, multidim=self.sol.neigs*4, name='modes_raw')
                    self.sol.ez = ngs.GridFunction(self.fes, multidim=self.sol.neigs*4, name='modes_raw')

                    for i in range(min(nconvergedTM, neigs_query)):
                        eigval = self._QTM.getEigenpair(i, petsc_yTM)
                        tau_left = eigval - delta2/(4*eigval)
                        tau_right = eigval + delta2/(4*eigval)
                        kappa2 = 0.5*Sigma2-eigval**2-delta2**2/(16*eigval**2)

                        self.sol.beta_TM[i] = np.sqrt(kappa2)
                        self.sol.hz.vecs[i].data = petsc_yTM.array
                        eigvals_TM[i] = eigval

                    for i in range(min(nconvergedTE, neigs_query)):
                    # for i in range(min(nconvergedTE, self.sol.neigs)):
                        eigval = self._QTE.getEigenpair(i, petsc_yTE)
                        tau_left = eigval - delta2/(4*eigval)
                        tau_right = eigval + delta2/(4*eigval)
                        kappa2 = Sigma2 - eigval**2 -delta2**2/(16*eigval**2)

                        self.sol.beta_TE[i] = np.sqrt(kappa2)
                        self.sol.ez.vecs[i].data = petsc_yTE.array
                        eigvals_TE[i] = eigval

                #### SLEPc.NEP()
                # else:

                #     f0 = SLEPc.FN().create()
                #     f0.setType(SLEPc.FN.Type.RATIONAL)
                #     f0.setRationalNumerator([1.0])

                #     f1 = SLEPc.FN().create()
                #     f1.setType(SLEPc.FN.Type.RATIONAL)
                #     f1.setRationalNumerator([1.0, 0.0])

                #     kx2_left = SLEPc.FN().create()
                #     kx2_left.setType(SLEPc.FN.Type.RATIONAL)
                #     kx2_left.setRationalNumerator([-1.0, self._k2left])

                #     kx2_right = SLEPc.FN().create()
                #     kx2_right.setType(SLEPc.FN.Type.RATIONAL)
                #     kx2_right.setRationalNumerator([-1.0, self._k2right])

                #     f_sqrt = SLEPc.FN().create()
                #     f_sqrt.setType(SLEPc.FN.Type.SQRT)

                #     kx_left = SLEPc.FN().create()
                #     kx_left.setType(SLEPc.FN.Type.COMBINE)
                #     kx_left.setCombineChildren(SLEPc.FN.CombineType.COMPOSE, f_sqrt, kx2_left)

                #     kx_right = SLEPc.FN().create()
                #     kx_right.setType(SLEPc.FN.Type.COMBINE)
                #     kx_right.setCombineChildren(SLEPc.FN.CombineType.COMPOSE, f_sqrt, kx2_right)

                #     self._NEPTM = SLEPc.NEP().create()
                #     self._NEPTM.setSplitOperator(petsc_AListTM, [f0, f1, kx_left, kx_right], PETSc.Mat.Structure.SUBSET)
                #     # self._NEPTM.setSplitOperator(petsc_AListTM[:-2], [f0, f1], PETSc.Mat.Structure.SUBSET) # test
                #     self._NEPTM.setTolerances(1.0e-12, 200)
                #     self._NEPTM.setDimensions(neigs_query) # number of requested eigenvalues
                #     # self._NEPTM.setProblemType(SLEPc.NEP.ProblemType.RATIONAL)
                #     self._NEPTM.setType(SLEPc.NEP.Type.CISS)
                #     # self._NEPTM.setType(SLEPc.NEP.Type.NLEIGS)

                #     # eigenvalues in a contour
                #     # only the elliptic region is implemented in SLEPc.NEP
                #     rg = self._NEPTM.getRG()
                #     rg.setType(SLEPc.RG.Type.ELLIPSE)
                #     rg.setEllipseParameters(k2max, np.abs(k2max), 1)

                #     self._NEPTM.setFromOptions()
                #     # Solve the eigensystem
                #     self._NEPTM.solve()
                #     # self._NEPTM.valuesView()

                #     self._NEPTE = SLEPc.NEP().create()
                #     self._NEPTE.setSplitOperator(petsc_AListTE, [f0, f1, kx_left, kx_right], PETSc.Mat.Structure.SUBSET)
                #     # self._NEPTE.setSplitOperator(petsc_AListTE[:-2], [f0, f1], PETSc.Mat.Structure.SUBSET) # test
                #     self._NEPTE.setTolerances(1.0e-12, 200)
                #     self._NEPTE.setDimensions(neigs_query) # number of requested eigenvalues
                #     self._NEPTE.setType(SLEPc.NEP.Type.CISS)
                #     # self._NEPTE.setType(SLEPc.NEP.Type.NLEIGS)

                #     rg = self._NEPTE.getRG()
                #     rg.setType(SLEPc.RG.Type.ELLIPSE)
                #     rg.setEllipseParameters(k2max, np.abs(k2max), 1)

                #     self._NEPTE.setFromOptions()
                #     # Solve the eigensystem
                #     self._NEPTE.solve()

                #     # self._NEPTE.valuesView()

                #     petsc_y = petsc_AListTM[0].createVecs(side='right')
                #     # For saving solution eigen vectors
                #     self.sol.hz = ngs.GridFunction(self.fes, multidim=self.sol.neigs*4, name='modes_raw')
                #     self.sol.ez = ngs.GridFunction(self.fes, multidim=self.sol.neigs*4, name='modes_raw')

                #     nconvergedTM = self._NEPTM.getConverged()
                #     eigvals_TM = np.empty((nconvergedTM), dtype=np.complex128)
                #     # if nconvergedTM > 0:
                #     for i in range(min(nconvergedTM, neigs_query)):
                #     # for i in range(min(nconvergedTM, self.sol.neigs)):
                #         eigval = self._NEPTM.getEigenpair(i, petsc_y)
                #         self.sol.beta_TM[i] = np.sqrt(eigval)
                #         self.sol.hz.vecs[i].data = petsc_y.array
                #         eigvals_TM[i] = eigval

                #     nconvergedTE = self._NEPTE.getConverged()
                #     eigvals_TE = np.empty((nconvergedTE), dtype=np.complex128)
                #     # if nconvergedTE > 0:
                #     for i in range(min(nconvergedTE, neigs_query)):
                #     # for i in range(min(nconvergedTE, self.sol.neigs)):
                #         eigval = self._NEPTE.getEigenpair(i, petsc_y)
                #         self.sol.beta_TE[i] = np.sqrt(eigval)
                #         self.sol.ez.vecs[i].data = petsc_y.array
                #         eigvals_TE[i] = eigval

                # drived quantities
                neff_TM = self.sol.beta_TM[:nconvergedTM]/self.k0
                neff_TE = self.sol.beta_TE[:nconvergedTE]/self.k0
                self.sol.neff_TM[:nconvergedTM] = neff_TM
                self.sol.neff_TE[:nconvergedTE] = neff_TE

                # self.sol.epsilon_TM = self.sol.neff_TM**2
                # self.sol.epsilon_TE = self.sol.neff_TE**2
                self._eigvals_TM = eigvals_TM
                self._eigvals_TE = eigvals_TE
                # self._vertices = np.array(vertices)
                # self._neff_vertices = np.sqrt(0.5*Sigma2-self._vertices**2-delta2**2/(16*self._vertices**2))

        #%% solve w/o TBC
        #
        # spcipy.sparse.linalg provides eigs and eigsh which are wrapped around
        # the ARPACK eigenvalue solvers.
        # https://github.com/scipy/scipy/blob/v1.11.2/scipy/sparse/linalg/_eigen/arpack/arpack.py
        #
        else:
            # for linear eigenvalue problem
            if method=="ngsolve_ArnoldiSolver":
                self._aTM.Assemble()
                self._mTM.Assemble()
                self._A0TE.Assemble()
                self._mTE.Assemble()

                # inspect the matrix pattern
                if show_pattern:
                    fig, axs = _plt.subplots(figsize = (2*_fig.h*_fig.inch2cm*_fig.zoom, 1.86*_fig.h*_fig.inch2cm*_fig.zoom), ncols = 2, nrows = 2, constrained_layout = True);

                    arows, acols, avals = self._aTM.mat.COO()
                    mrows, mcols, mvals = self._mTM.mat.COO()
                    ATM = scipy_sparse.csr_matrix( (avals, (arows, acols)) )
                    MTM = scipy_sparse.csr_matrix( (mvals, (mrows, mcols)) )

                    arows, acols, avals = self._A0TE.mat.COO()
                    mrows, mcols, mvals = self._mTE.mat.COO()
                    ATE = scipy_sparse.csr_matrix( (avals, (arows, acols)) )
                    MTE = scipy_sparse.csr_matrix( (mvals, (mrows, mcols)) )

                    # condition number
                    ATMdense = ATM.todense()
                    MTMdense = MTM.todense()
                    ATEdense = ATE.todense()
                    MTEdense = MTE.todense()

                    # axs[0,0].spy(ATM);
                    # axs[1,0].spy(MTM);
                    # axs[0,1].spy(ATE);
                    # axs[1,1].spy(MTE);

                    axs[0,0].imshow(np.abs(ATMdense));
                    axs[1,0].imshow(np.abs(MTMdense));
                    axs[0,1].imshow(np.abs(ATEdense));
                    axs[1,1].imshow(np.abs(MTEdense));

                    axs[0,0].set_title("K($A_{\\mathrm{TM}}$) ="+"{0.real:.2f}".format(np.linalg.cond(ATMdense, p='fro')))
                    axs[1,0].set_title("K($M_{\\mathrm{TM}}$) ="+"{0.real:.2f}".format(np.linalg.cond(MTMdense, p='fro')))
                    axs[0,1].set_title("K($A_{\\mathrm{TE}}$) ="+"{0.real:.2f}".format(np.linalg.cond(ATEdense, p='fro')))
                    axs[1,1].set_title("K($M_{\\mathrm{TE}}$) ="+"{0.real:.2f}".format(np.linalg.cond(MTEdense, p='fro')))

                    _plt.show();

                    # This reveals that only part of the dofs are linked to nodal points, which is useful for plotting the field profile
                    for i in range(self.fes.ndof):
                        print (i,":", self.fes.CouplingType(i))
                    # return

                # For saving solution eigen vectors
                hz = ngs.GridFunction(self.fes, multidim=self.sol.neigs, name='modes_raw')
                ez = ngs.GridFunction(self.fes, multidim=self.sol.neigs, name='modes_raw')
                # hz_sorted = ngs.GridFunction(self.fes, multidim=self.sol.neigs, name='modes_sorted')
                # ez_sorted = ngs.GridFunction(self.fes, multidim=self.sol.neigs, name='modes_sorted')

                #### Search eigenvalues around eig_sigma.
                # pro- and post-processing for this shift is done internally in the eigenvalue solver
                self.sol._eig_sigma = 1*self.k0**2 # search around the vacuum light line

                with ngs.TaskManager():
                    eigvalsTM = ngs.ArnoldiSolver(self._aTM.mat, self._mTM.mat, self.fes.FreeDofs(), list(hz.vecs), shift=self.sol._eig_sigma)
                    eigvalsTE = ngs.ArnoldiSolver(self._A0TE.mat, self._mTE.mat, self.fes.FreeDofs(), list(ez.vecs), shift=self.sol._eig_sigma)

                    #### save to .sol node
                    self.sol.neff_TM = np.sqrt(eigvalsTM)
                    self.sol.neff_TE = np.sqrt(eigvalsTE)

                    self.sol.beta_TM = self.sol.neff_TM*self.k0
                    self.sol.beta_TE = self.sol.neff_TE*self.k0

                    self.sol.epsilon_TM = self.sol.neff_TM**2
                    self.sol.epsilon_TE = self.sol.neff_TE**2

                    self.sol.hz = hz
                    self.sol.ez = ez

    #### init
    def __init__(self, w=1.0, intervals=(0, 1), nnodes=(17, 0), labels=("freespace", "dummy")):

        self._ngsolve_w.Set(w)

        self.w = self._ngsolve_w.Get()

        self.geom = self.Geometry(intervals, nnodes, labels)

        # The mesh creation needs the information from geometry, so a reference to geom is given as an argument.
        self.mesh = self.Mesh(self.geom)

        self.material = self.Material(self)

        self.figure = self.Figure(self)

        self.TBC = False

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
    _ngsolve_w = ngs.Parameter(1.0)
    # # @property # read-only, but not work with ngsolve, cannot be overwritten into other type or value
    # def _ngsolve_w(self): # prefix "_" indicates a private variable
        # return ngs.Parameter(self.w) # Later get updated by _ngsolve_w.Set(w)

    _ngsolve_ld0_target = ngs.Parameter(780e-9)

    # Interdependent variables, that can be modified via either one of them.
    # Spectral information, everything is anchored to the normalized frequency.
    # w: normalized frequency ld0_target/ld0 and k0 = 2*np.pi/ld0, with d_target==ld0_target
    w = _w_access()
    ld0 = _ld0_access()
    k0 = _k0_access()

    # a range of normalized frequencies
    # wArray = np.arange(0.45, 2.45, 0.01)
    wArray = np.array([1.0])
    ld0Array = _ld0Array_access()

    # The target wavelength, independent variable
    ld0_target = _ld0_target_access()

    # Derived ngsolve CoefficientFunctions for being used
    # by other ngsolve CoefficientFunctions.
    _ngsolve_omega = 2*np.pi*const_c*_ngsolve_w/_ngsolve_ld0_target
    _ngsolve_k0 = 2*np.pi/(_ngsolve_ld0_target/_ngsolve_w)

# __all__ = ["SlabWaveGuide", "_fig", "const"]
