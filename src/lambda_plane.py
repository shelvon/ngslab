#!/usr/bin/env python3
"""
@author: shelvon
@email: xiaorun.zang@outlook.com

"""

# If nklist[0].real != nklist[-1].real and nklist[0].imag != nklist[-1].imag,
# then kappa values in the first quadrant for forward propagating and decaying modes are distributed in 9 regions, which are divided by two horizontal and two vertical straight lines passing through the two points: nklist[0] and nklist[-1] in the complex plane.

def plotRegions(axs=None, nklist=[1.5, 2.0, 1.0], kappa_modes=None):
    import sys
    import numpy as np

    # optical properties
    nbc = np.array([nklist[0], nklist[-1]]) # force to be a numpy array
    # nbc[0]: refractive index of the substrate
    # nbc[1]: refractive index of the cover

    nbc_flip = False
    if nbc[0].real<nbc[1].real:
        nbc = nbc[::-1]
        nbc_flip = True

    n2bc = nbc**2

    # sampling in each kappa2 region
    nx = 31 # number of the sampling points along kappa2.real
    ny = 31 # number of the sampling points along kappa2.real
    offset = 1e-6 # a small shift from the branch cuts, or branch points

    # the branch cuts are vertical lines going through
    cut_by_kappa2 = True # kappa2.real
    cut_by_kappa2 = False # kappa.real

    cut_by_kappa = not cut_by_kappa2

    debugging = True
    debugging = False
    if not debugging:
        # more refined sampling points
        nx *= 3
        ny *= 3

    n_re_max = max(np.abs(nklist))*1.5
    n_re_min = 0
    n_im_max = n_re_max
    n_im_min = 0
    n2_re_max = n_re_max**2
    n2_im_max = n2_re_max*np.sqrt(2)

    n_re_step = (n_re_max-n_re_min)/nx
    n_im_step = (n_im_max-n_im_min)/ny

    # zf_lambda = 1.0
    # xmin = np.array([n_re_min, n_re_min*zf_lambda, n_re_min, n_re_min])
    # xmax = np.array([n_re_max, n_re_max*zf_lambda, n_re_max, n_re_max])
    # ymin = np.array([n_im_min, n_im_min*zf_lambda, n_re_min, n_re_min])
    # ymax = np.array([n_im_max, n_im_max*zf_lambda, n_re_max, n_re_max])

    # xlabels = ["$\\kappa '$", "$\\lambda '$", "$\\tau_s'$", "$\\tau_c'$"]
    # ylabels = ["$\\kappa ''$", "$\\lambda ''$", "$\\tau_s''$", "$\\tau_c''$"]
    xlabels = ["${(\\kappa^{2})} '$", "$\\lambda '$", "$\\tau_\mathrm{s}'$", "$\\tau_\mathrm{c}'$"]
    ylabels = ["${(\\kappa^{2})} ''$", "$\\lambda ''$", "$\\tau_\mathrm{s}''$", "$\\tau_\mathrm{c}''$"]
    #%% plotting parameters
    fontsize = 8
    ms = 10 # markersize
    # fill_modes = "full";
    # alpha_modes = 1.0;
    # alpha = 0.6

    tau_regions = [["", "", ""], ["", "B", "A"], ["", "C", "D"]]
    for i in range(4):
        # labels
        axs[i].text(0.02, 0.02, '('+chr(i+97)+')', transform=axs[i].transAxes, fontsize=fontsize*1.25, ha="left", va="bottom", fontweight="book")

        # axs[i].set_xlabel(xlabels[i], usetex=True, size=fontsize*1.25, x=1, labelpad=-10, transform=axs[i].transAxes)
        # axs[i].set_ylabel(ylabels[i], usetex=True, size=fontsize*1.25, rotation=0, y=1.0, labelpad=-10, va="bottom", ha="left", transform=axs[i].transAxes)
        axs[i].text(1.0, -0.08, xlabels[i], usetex=True, transform=axs[i].transAxes, fontsize=fontsize*1.25, ha="center", va="top", fontweight="book")
        axs[i].text(0.0, 1.10, ylabels[i], usetex=True, transform=axs[i].transAxes, fontsize=fontsize*1.25, ha="right", va="center", fontweight="book")

        for iSx in [+1, -1]:
            for itau_im in [+1, -1]:
                axs[2].text(0.5+iSx*0.25, 0.5+itau_im*0.25,
                            "$\\mathrm{"+tau_regions[-itau_im][-iSx]+"}_\mathrm{s}$",
                            transform=axs[2].transAxes,
                            color="k", ha="center", va="center", fontsize=fontsize*1.25,
                            )
                axs[3].text(0.5+iSx*0.25, 0.5+itau_im*0.25,
                            "$\\mathrm{"+tau_regions[itau_im][iSx]+"}_\mathrm{c}$",
                            transform=axs[3].transAxes,
                            color="k", ha="center", va="center", fontsize=fontsize*1.25,
                            )
        if i in [0, 1]:
            if nbc_flip:
                n2 = n2bc[1-i]
            else:
                n2 = n2bc[1-i]
            axs[2+i].axline((0, 0), slope=0, linestyle="--", linewidth=1, c="k")
            if np.abs(n2.real)<1e-16:
                axs[2+i].axline((0, 0), slope=np.divide(-n2.real, n2.imag), linestyle="--", linewidth=1, c="k")
            else:
                axs[2+i].axline((0, 0), (-n2.imag/n2.real, 1), linestyle="--", linewidth=1, c="k")

    axs[1].axline((0, 0), slope=0, linestyle="--", linewidth=1, c="k")
    axs[1].axline((0, 0), (0, 1), linestyle="--", linewidth=1, c="k")

    #%% branch cuts

    cutc = ["k", "k"] # color of branch cut
    def _plotCut(grid, bc):
        for icut in range(grid[:,0].size):

            # branch cuts in epsilon plane
            axs[0].plot(grid[icut,:].real, grid[icut,:].imag, "-", linewidth=1.5, c=cutc[icut], ms=ms)

            # mark the media in the complex n and epsilon planes
            markerstyle = dict(marker="o", markeredgewidth=0, fillstyle="full", markerfacecolor='w', ms=5, zorder=1e4, clip_on=False)
            axs[0].plot(bc[icut].real, bc[icut].imag, c=cutc[icut], **markerstyle)

            # markerstyle = dict(marker="x", markeredgewidth=1, fillstyle="full", ms=3, zorder=1e6, clip_on=False, markerfacecolor='w')
            # axs[0].plot(bc[icut].real, bc[icut].imag, c=cutc[icut], **markerstyle)

    zf_cut_im = 1
    if cut_by_kappa2:
        kappa2_cut = np.array(
            [np.linspace(n2bc[0], n2bc[0] + 1j*n2_im_max*zf_cut_im, ny),
             np.linspace(n2bc[1], n2bc[1] + 1j*n2_im_max*zf_cut_im, ny)
             ])
        kappa_cut = kappa2_cut**0.5
        _plotCut(kappa2_cut, n2bc)
        # _plotCut(kappa_cut, nbc)

    else:
        kappa_cut = np.array(
            [np.linspace(nbc[0], nbc[0] + 1j*n_im_max*zf_cut_im, ny),
             np.linspace(nbc[1], nbc[1] + 1j*n_im_max*zf_cut_im, ny)
              ])
        kappa2_cut = kappa_cut**2
        # _plotCut(kappa_cut, nbc)
        _plotCut(kappa2_cut, n2bc)

    #%% definition regions for kappa2 values
    def _createGrid(x0, x1, y0, y1, x_step=n_re_step, y_step=n_im_step, fill_offset=1):

        if (x0+offset) < (x1-offset):
            nx = int(np.ceil((x1-x0)/x_step))
            x = np.linspace(x0+offset, x1-offset, nx)
            x = np.pad(x, (1, 1), "constant", constant_values=(x0+0.5*offset, x1-0.5*offset))
        elif x0<=x1:
            x = np.array([0.5*(x0+x1)])
        else:
            print("x1 cannot be smaller than x0!"); sys.exit(0);

        if (y0+offset) < (y1-offset):
            ny = int(np.ceil((y1-y0)/y_step))
            y = np.linspace(y0+offset, y1-offset, ny)
            y = np.pad(y, (1, 1), "constant", constant_values=(y0+0.5*offset, y1-0.5*offset))
        elif y0<=y1:
            y = np.array([0.5*(y0+y1)])
        else:
            print("y1 cannot be smaller than y0!"); sys.exit(0);

        # grid points
        xx, yy = np.meshgrid(x, y, indexing="xy")
        zz = xx + 1j*yy

        # grid values for plotting the region with specified filling colors
        cc = yy**0 + fill_offset-1 # a uniform color
        # cc = yy/max(offset, np.max(np.abs(yy))) + fill_offset-1

        # print(np.min(np.abs(cc)), np.max(np.abs(cc)))

        ## this is important for tricontourf, different value (and color) at the boundaries
        cc[[0, -1], :] = 0
        cc[:, [0, -1]] = 0

        return zz, cc

    # points = [p1, p2] are two points (x) in the 1st quadrant of the complex plane
    # In general, they divide the 1st quadrant into 9 regions as follows.
    #
    # ^
    # |                   y_max
    # |     |     |
    # | (9) | (6) |  (3)
    # |-----|-----x------ y1
    # | (8) | (5) |  (2)
    # |-----x-----|------ y0
    # | (7) | (4) |  (1)
    # |---------------------->
    #       x0    x1      x_max
    def _createRegions(points, x_max, y_max, x_min=0, y_min=0):
        x0, x1 = np.sort(np.asarray(points).real)
        y0, y1 = np.sort(np.asarray(points).imag)
        # print(x0, x1, y0, y1)

        grid_1, zgrid_1 = _createGrid(x1, x_max, y_min, y0, n_re_step, n_im_step/5, 1)
        grid_2, zgrid_2 = _createGrid(x1, x_max, y0, y1, n_re_step, n_im_step/2, 2)
        grid_3, zgrid_3 = _createGrid(x1, x_max, y1, y_max, n_re_step, n_im_step, 3)

        grid_4, zgrid_4 = _createGrid(x0, x1, y_min, y0, n_re_step, n_im_step/5, 5)
        grid_5, zgrid_5 = _createGrid(x0, x1, y0, y1, n_re_step, n_im_step/2, 6)
        grid_6, zgrid_6 = _createGrid(x0, x1, y1, y_max, n_re_step, n_im_step, 7)

        grid_7, zgrid_7 = _createGrid(x_min, x0, y_min, y0, n_re_step, n_im_step/5, 7)
        grid_8, zgrid_8 = _createGrid(x_min, x0, y0, y1, n_re_step, n_im_step/2, 8)
        grid_9, zgrid_9 = _createGrid(x_min, x0, y1, y_max, n_re_step, n_im_step, 9)

        # print(grid_9.shape, grid_6.shape, grid_3.shape)
        # print(grid_8.shape, grid_5.shape, grid_2.shape)
        # print(grid_7.shape, grid_4.shape, grid_1.shape)

        zz = np.block([[grid_9, grid_6, grid_3],
                        [grid_8, grid_5, grid_2],
                        [grid_7, grid_4, grid_1]])

        cc = np.block([[zgrid_9, zgrid_6, zgrid_3],
                        [zgrid_8, zgrid_5, zgrid_2],
                        [zgrid_7, zgrid_4, zgrid_1]])

        return zz, cc

    if cut_by_kappa:
        kappagrid, ccgrid = _createRegions(nbc, n_re_max, n_im_max)
        kappa2grid = kappagrid**2
    else:
        kappa2grid, ccgrid = _createRegions(n2bc, n2_re_max, n2_im_max, -n2_re_max)
        kappagrid = kappa2grid**0.5

    # sign_kxleft = np.array([-1, 1, 1, 1, -1, -1])
    # sign_kxright = np.array([-1, -1, 1, 1, 1, -1])

    # get kx values in the right Riemann's sheet for different n2 and kappa2 values
    def _get_kx(n2, kappa2grid, cut_by_kappa2=True):
        kxleft = (n2-kappa2grid)**0.5
        if cut_by_kappa2:
            signs = np.array([-1, 1, ])
            # divide into 4 quadrants centered at n2
            idx_q1 = np.where(
                (kappa2grid.real>n2.real) & (kappa2grid.imag>n2.imag))[0]

            idx_q2 = np.where(
                (kappa2grid.real<n2.real) & (kappa2grid.imag>n2.imag))[0]

            idx_q3 = np.where(
                (kappa2grid.real<n2.real) & (kappa2grid.imag<n2.imag))[0]

            idx_q4 = np.where(
                (kappa2grid.real>n2.real) & (kappa2grid.imag<n2.imag))[0]
        else:
            # divide into 4 quadrants centered at n
            #
            # Since n and kappa are in the first quadrant, then
            # n2 and kappa2 is always in the upper half space of the complex plane.
            # Taking the square root directly is totally safe.
            n = n2**0.5
            kappagrid = kappa2grid
            # quadrant 1
            idx_q1 = np.where(
                (kappagrid.real>n.real) & (kappagrid.imag>n.imag))[0]

            # quadrant 2
            idx_q2 = np.where(
                (kappagrid.real<n.real) & (kappagrid.imag>n.imag))[0]

            # quadrant 3
            idx_q3 = np.where(
                (kappagrid.real<n.real) & (kappagrid.imag<n.imag))[0]

            # quadrant 4
            idx_q4 = np.where(
                (kappagrid.real>n.real) & (kappagrid.imag<n.imag))[0]

        # return kxleft
    kxleft = _get_kx(n2bc[0], kappa2grid, cut_by_kappa2)

    # sys.exit(0)

    kxleft = (n2bc[0]-kappa2grid)**0.5
    kxright = (n2bc[1]-kappa2grid)**0.5

    # for ir in range(3):
    #     kxleft[ir] *= sign_kxleft[ir]
    #     kxright[ir] *= sign_kxright[ir]

    # lambda values
    lambdagrid = np.array([0.5*(kxleft + kxright),
                           0.5*(-kxleft + kxright),
                           0.5*(kxleft - kxright),
                           0.5*(-kxleft - kxright)])

    sign_cases = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])

    # hatches = np.array(["++++", "////", "\\\\", "----"])
    #%% plot the definition regions
    # levels_cmap = np.array([0, 1]) # just two colors
    # levels_cmap = np.linspace(0.1, 10.0, 21)
    # levels_cc = np.linspace(0.1, 10.1, 21)
    levels_cc = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) +0.1
    # cmap = "hsv"
    # cmap = "Wistia"
    cmap = "twilight_shifted"
    # cmaps = ["Blues", "Oranges", "Purples", "Greys", "Reds", "Greens"] # color of each region

    alpha = 0.3
    # print(np.min(ccgrid), np.max(ccgrid))

    # markerstyle = dict(marker="<", s=28, alpha=alpha_modes, zorder=1e8, clip_on=False)
    # axs[0].scatter(kappa_modes.real, kappa_modes.imag, c="gray", **markerstyle)

    # for ir in range(np.size(zgrid, 0)):
    for ir in range(1):
        # cmap = cmaps[ir]
        # axs[0].contourf(kappagrid.real, kappagrid.imag, ccgrid, cmap=cmap, levels=levels_cc)
        axs[0].contourf(kappa2grid.real, kappa2grid.imag, ccgrid, cmap=cmap, levels=levels_cc, alpha=alpha)

        continue # skip plotting regions in the following complex planes
        # in the lambda plane
        for icase in [0]:
            axs[1].tricontourf(lambdagrid[icase].real.flatten(), lambdagrid[icase].imag.flatten(), ccgrid.flatten(), cmap=cmap, levels=levels_cc, alpha=alpha)

            # in the kxleft and kxright planes
            axs[2].tricontourf(sign_cases[icase, 0]*kxleft.real.flatten(),
                               sign_cases[icase, 0]*kxleft.imag.flatten(),
                               ccgrid.flatten(), cmap=cmap, levels=levels_cc, alpha=alpha)

            axs[3].tricontourf(sign_cases[icase, 1]*kxright.real.flatten(),
                               sign_cases[icase, 1]*kxright.imag.flatten(),
                               ccgrid.flatten(), cmap=cmap, levels=levels_cc, alpha=alpha)

    # for i in [1, 2, 3]:
    #     axs[i].set_aspect("equal", "datalim")


# test
if __name__ == "__main__":
    import matplotlib.pyplot as _plt
    inch2cm = 1.0/2.54  # inches-->centimeters
    zf = 2.0
    figsize = [zf*8.6*inch2cm, zf*6.4*inch2cm]
    _plt.close("all")

    fig, _axs = _plt.subplots(
        figsize = figsize, ncols = 2, nrows = 2, constrained_layout = False);
    axs = _axs.flatten()

    # plotRegions(axs, nklist=[1.5+0.5j, 1.6+1.0j, 1.0+0.3j])
    plotRegions(axs, nklist=[0.1+5.5j, 1.6+1.0j, 1.0+0.01j])
