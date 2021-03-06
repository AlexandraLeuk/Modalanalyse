# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:39:26 2017

@author: Sebastian
"""

from scipy import sparse
from scipy.io import mmread
import os
import numpy as np
import matplotlib.pyplot as plt
from JSAnimation import IPython_display
from matplotlib import animation
from matplotlib.pyplot import subplots
# import misc


class FEMSys(object):
    def __init__(self):
        WD = os.getcwd()
        fstr = WD + '/data/{}'

        self.M = sparse.csc_matrix(mmread(fstr.format('Ms.mtx')))
        self.K = sparse.csc_matrix(mmread(fstr.format('Ks.mtx')))
        self.C = 0.1 * self.M
        self.X = mmread(fstr.format('X.mtx'))

        self.nN = self.X.shape[0]
        self.nDof = 3 * self.nN

        # Default is unbounded
        self.Ic = np.zeros(self.nDof, dtype=bool)
        # C = sparse.csc_matrix(K.shape) # a zeros damping matrix
        # N = X.shape[0] # number of nodes

    def set_boundary(self, fix=None):
        self.Ic = np.zeros(self.nDof, dtype=bool)

        if fix is not None:
            self.Ic |= fix

        Ic = self.Ic
        self.Kc = sparse.csc_matrix(self.K[np.ix_(~Ic, ~Ic)])
        self.Mc = sparse.csc_matrix(self.M[np.ix_(~Ic, ~Ic)])
        self.Cc = sparse.csc_matrix(self.C[np.ix_(~Ic, ~Ic)])
        # Cc = sparse.csc_matrix(C[np.ix_(~Ic,~Ic)])

    def sim_freq(self, mode_count=1):
        Mc = self.Mc
        Kc = self.Kc

        Wc, Vc = sparse.linalg.eigsh(Kc, mode_count, Mc, sigma=0.0)
        V = np.zeros([self.nDof, mode_count])
        V[~self.Ic, :] = Vc
        w0 = np.sqrt(Wc)/2/np.pi

        res = FEMRes(self)
        res.x = V
        res.w0 = w0
        return res

    def sim_steady_state(self, w_list, force):
        xw = np.zeros((self.nDof, len(w_list)), dtype=complex)
        for i, w in enumerate(w_list):
            Zw = self.Kc + 1j*2*np.pi*w*self.Cc - (2*np.pi*w)**2 * self.Mc
            xw[~self.Ic, i] = sparse.linalg.spsolve(Zw, force[~self.Ic])
        res = FEMRes(self)
        res.x = xw
        return res

    def sim_static(self, f):
        u = np.zeros(self.nDof)
        u[~self.Ic] = sparse.linalg.spsolve(self.Kc,f[~self.Ic])
        res = FEMRes(self)
        res.x = u
        return res

    def sim_time(self, t, force=None, gamma=.5, beta=.25):
        # Assume equal spaced time discretization
        dt = t[1] - t[0]

        # Newmark parameters
        a0 = 1. / (beta * dt**2)
        a1 = gamma / (beta * dt)
        a2 = 1. / (beta * dt)
        a3 = 1. / (2. * beta) - 1.
        a4 = gamma / beta - 1.
        a5 = dt / 2. * (gamma / beta - 2.0)
        a6 = dt * (1.0 - gamma)
        a7 = gamma * dt

        Mc = self.Mc
        Kc = self.Kc
        Cc = self.Cc
        Kn = Kc + a0*Mc + a1*Cc
        solve_step = sparse.linalg.factorized(Kn)

        # initialize (in the constraint system)
        ut = np.zeros([np.sum(~self.Ic)])  # initial displacement
        vt = np.zeros_like(ut)  # initial velocity
        at = np.zeros_like(ut)  # initial acceleration

        # preallocate result ( in full system )
        x = np.zeros([self.nDof, len(t)])

        # do the time integration
        for i in range(len(t)):
            # compute effective forcing vector
            ft1 = force[~self.Ic, i] + Mc* \
                (a0*ut + a2*vt + a3*at) + Cc*(a1*ut + a4*vt + a5*at)
            # solve for u at t+dt using the pre-factorized matrix
            ut1 = solve_step(ft1)
            # update v & a
            at1 = a0*(ut1-ut) - a2*vt - a3*at
            vt1 = vt + a6*at + a7*at1
            # save and prepare next step
            x[~self.Ic, i] = ut1
            ut = ut1
            vt = vt1
            at = at1

        res = FEMRes(self)
        res.x = x
        res.t = t
        return res


class FEMRes(object):
    def __init__(self, fem_sys):
        self.sys = fem_sys
        self.X = fem_sys.X
        self.nDof = fem_sys.nDof
        nprec = 6 # precision for finding uniqe values
        # get grid vectors (the unique vectors of the x,y,z coodinate-grid)
        self.xv = np.unique(np.round(self.X[:,0],decimals=nprec))
        self.yv = np.unique(np.round(self.X[:,1],decimals=nprec))
        self.zv = np.unique(np.round(self.X[:,2],decimals=nprec))

    def reduce_system(self, f):
        V = self.x
        K = self.sys.K.toarray()
        M = self.sys.M.toarray()

        self.K_red = V.T @ K @ V
        self.M_red = V.T @ M @ V
        self.f_red = V.T @ f

        self.u_red = np.linalg.solve(self.K_red,self.f_red) # compute displacement in new basis
        self.u_proj = V @ self.u_red
        self.u_proj = np.reshape(self.u_proj, self.nDof)

    def compute_MPF(self, e):
        V = self.x
        n_modes = V.shape[1]
        MPF = np.zeros(n_modes)
        for i, v in enumerate(V.T): # iterate over eigenvectors
            MPF[i] = np.abs(np.dot(v, self.sys.M.dot(e)) / np.dot(v, self.sys.M.dot(e)))
            # MPF[i] = np.abs(v @ self.sys.M.toarray() @ e) / (v @ self.sys.M.toarray() @ e)
        return MPF, V

    def plot_MPF(self, e, title=''):
        MPF = self.compute_MPF(e)
        x = range(len(MPF))
        width = 0.75
        fig, ax = plt.subplots()
        ax.bar(x, MPF, width, color="blue")
        ax.set_xlabel('Measure of coincidance')
        ax.set_ylabel('Mode index')
        # ax.title
        # plt.title(title)
        # plt.show()
        return fig, ax

    def plot_dof(self, dof, xlabel='Time in s', ylabel='displacement in m'):
        fig, ax = plt.subplots()
        ax.plot(self.t, self.x[dof, :].T)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig, ax

    def plot_2d(self, id_plane, system='original'):
        if system == 'original':
            if len(self.x.shape) == 1:
                u = np.reshape(self.x, (self.nDof, 1))
            else:
                u = self.x
        elif system == 'reduced':
            u = np.reshape(self.u_proj, (self.nDof, 1))

        for i, v in enumerate(u.T):
            c = np.reshape(v[id_plane],[len(self.yv),len(self.xv)])
            lim = np.max(np.abs(c))
            fig,ax = plt.subplots(figsize=[3.5,2])
            ax.contourf(self.xv,self.yv,c,cmap=plt.get_cmap('RdBu'),vmin=-lim,vmax=lim)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
        return fig, ax

    def animate(self, ID, idx=0):
        """
        This function will use one mode shape vector of global DoFs,
        select the z-dofs for nodes on the top surface (Nt) and return an
        animation of contour plots.
        """

        u = self.x[:, idx]
        u = u[ID]

        # for plotting
        nprec = 6  # precision for finding uniqe values
        # get grid vectors (the unique vectors of the x,y,z coodinate-grid)
        xv = np.unique(np.round(self.X[:, 0], decimals=nprec))
        yv = np.unique(np.round(self.X[:, 1], decimals=nprec))
        # zv = np.unique(np.round(self.X[:, 2], decimals=nprec))

        c = np.reshape(u, [len(yv), len(xv)])

        # set limit
        lim = np.max(np.abs(c))

        # setup figure
        fig, ax = subplots(figsize=[5, 4])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # animate function is called for each 'frame' (input is frames[i])
        def animate(phi):
            # select the current data for time step i
            c_t = (c[:, :]*np.exp(1j*phi)).real
            cf = ax.contourf(xv, yv, c_t, cmap=plt.get_cmap(
                'RdBu'), vmin=-lim, vmax=lim)
            return cf

        # create animation (may take long for many frames)
        return animation.FuncAnimation(fig, animate, frames=np.linspace(0, 2*np.pi, 25, endpoint=False), interval=1000/25)

    # def animate_mode3d(self, mode=0):
    #     fig, ax = plt.subplots(subplot_kw={'projection':'3d'}) 

    #     # animate function is called for each 'frame' (input is frames[i])
    #     def animate(t):
    #         ax.cla()
    #         lim = .1
    #         ax.set_zlim([-lim, lim])
    #         ax.set_xlim([-3, 3])
    #         ax.set_ylim([-2, 2])
    #         u = np.real(self.x[:,k] * np.exp(self.w0[k]*2*np.pi*t*1j))
    #         cf = ax.scatter(self.X[:, 0]+u[B.x],self.X[:, 1]+u[B.y],beam.X[:, 2]+u[B.z],c='g',label='deformed')
    #         return cf

    #     # create animation (may take long for many frames)
    #     return animation.FuncAnimation(fig, animate, frames=np.linspace(0, 1/(res_harmonic.w0[k]), 15, endpoint=False), interval=1000/25)


class IdxMasks:
    def __init__(self, fem_sys):
        self.nDof = fem_sys.nDof

    # def new(self, include=None, exclude=None):
    #     if (include is not None ) and (exclude is None):
    #         pass

    def __setattr__(self, name, value):
        if type(value) in [np.ndarray, np.array, np.int32]:
            array = np.zeros(self.nDof, dtype=bool)
            array[value] = True
            super().__setattr__(name, array)
        else:
            super().__setattr__(name, value)


def fft(signal, t):
    # rfreq
    if len(signal.shape) == 2:
        if signal.shape[0] == 1:
            signal = signal[0, :]
        elif signal.shape[1] == 1:
            signal = signal[:, 0]

    N = len(t)
    dt = t[1] - t[0]

    amplitude = np.fft.fft(signal)
    freq = np.fft.fftfreq(N, dt)

    posfreq = freq >= 0
    amplitude=np.abs(amplitude[posfreq])
    freq = freq[posfreq]

    return freq, amplitude


if __name__ == '__main__':
    # & and, | or, ^ xor, ~ not
    # a = np.array([True, False, True])
    # b = np.array([True, True])
    # c = np.array([0, 1, 2])
    # print(c[c])
    # select nodes on the west-side, i.e. at x=x_min
    import sys
    print(sys.version_info)
    beam = FEMSys()
    B = IdxMasks(beam)