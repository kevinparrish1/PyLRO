import numpy as np
from ase.io import read, write
from ase.build.supercells import make_supercell
import matplotlib.pyplot as plt
from copy import copy
from plotly.figure_factory import create_trisurf
from colorsys import hsv_to_rgb
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.widgets import Slider







class amorphous_order:
    
    def __init__(self,filename,atom_number,fileformat='vasp'):
        """
        Args:
            filename: a crystal structure file
            atom_number: the atomic number of the element chosen for miller plane analysis. Only supports unit n=1 in unit cell
            fileformat: ASE accepted file format
        
        Returns:
            matplotlib 3dlpot of the miller sphere scaled to relative periodicities.
        """
        
        
        
        self.struc = read(filename,format=fileformat)
        self.atom_number=atom_number
        original = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.struc = make_supercell(self.struc, original) 
        if atom_number==None: #uses lowest atomic number species by default to calculate periodicity
            unique, counts = np.unique(self.struc.numbers, return_counts=True)
            atom_number=unique[np.argmin(counts)]
        idx=[i for i,x in enumerate(self.struc.numbers) if x==atom_number]
        points_=self.struc.get_scaled_positions() #grab scaled, <abc> independent positions
        
        self.atom_locations=np.array([points_[i] for i in idx])
        self.lattice_fit(atom_number)# Fits the structure to a lattice
        self.calculate_spacing()#Calculates spacings for a,b,c direction.

        

    def lattice_fit(self,n=8,x0=.5):
        """
        -Creates an nxnxn supercell of the shape of the structure (uses a,b,c vectors)
        -Uses optimization algorithm to fit the structure's atoms to this supercell of lattice points
        -Then calculates the <N_1,N_2,N_3> for each atom, where it should be in the lattice
        Args:
            n: the size of the sample supercell of lattice points
            x0: initial guess on how much to scale the supercell to fit to the structure's atoms
            
        Returns:
            
        """
        
        # cost function, calculates total distance between atoms and nearest lattice points
        def cost(scale,ll,al,lattice=False):
            ll_=copy(ll)
            ll_*=scale
            al_=copy(al)
            distance=[]
            lattice_match=[]
            ll_=[[i,x] for i,x in enumerate(ll_)]

            for a in al_:
                calc_=[]
                calc_=np.array([[x[0],np.linalg.norm(np.abs(np.array(x[1])-a))] for x in ll_])
                close_idx=np.argmin(calc_[:,1])
                distance.append(calc_[close_idx,1])
                lattice_match.append(int(calc_[close_idx,0]))
                ll_=list(ll_)
                del ll_[close_idx]


            if lattice:
                return np.average(distance), lattice_match
            else:
                return np.average(distance)
        
        #moves atom locations to origin
        al=copy(self.atom_locations)
        smallest=np.argmin([np.linalg.norm(x) for x in al])
        base=al[smallest]
        al=[x-al[smallest] for x in al]
        
        #creates sample of lattice points from structure. Is a normal sized lattice
        lattice_locations=[]
        norm_mat=self.struc.cell/np.linalg.norm(self.struc.cell)
        a,b,c=norm_mat[0],norm_mat[1],norm_mat[2]
        for ii in range(n):
            for jj in range(n):
                for kk in range(n):
                    lattice_locations.append(ii*a+jj*b+kk*c)
        lattice_locations=np.array(lattice_locations)

            
        #finds the scaling factor to fit to the structure's points
        self.scale=minimize(cost,x0,method='nelder-mead',args=(lattice_locations,al))['x']
        self.d,lattice_idx=cost(self.scale,lattice_locations,al,lattice=True)
        lattice_locations*=self.scale
        lattice_locations+=base

        
        #Solves for the i,j,k integers of each atom in the structure after the fit
        lattice_positions=np.zeros([len(self.atom_locations),3])
        for basis in range(3):
            v=lattice_locations[:,basis]
            vmin=np.min(v)
            v=np.array([x-vmin for x in v])
            v[v<.01]=0.
            vspace=np.min([x for x in v if x!=0.])
            v=[np.round(x/vspace) for x in v]
            
            v=[v[i] for i in lattice_idx]
            v=[x-np.min(v)+1 for x in v]
            lattice_positions[:,basis]=v
        
        self.lattice_points=lattice_positions
        self.ll=np.array([lattice_locations[i] for i in lattice_idx])





    
    def calculate_spacing(self):
        """
        Once the atoms are assigned to a lattice, the average scaled distance between calculated.
        Function normalizes the first points, then calculates average scaled spacing in a,b,c directions
        The spacing needs to be normalized so that planes in non basis directions can have relative deviations combined
        """
        a=copy(self.atom_locations)
        l=copy(self.lattice_points)
        l=np.array([[x[0]-1,x[1]-1,x[2]-1] for x in l])
        for i in range(3):
            mini=np.min(a[:,i])
            for j in range(len(a)):
                a[j][i]=a[j][i]-mini
        self.spacing=[]
        for basis in range(3):
            s=[]
            a_=a[:,basis]
            l_=l[:,basis]
            for j,x in enumerate(l_):
                if x!=0:
                    s.append(a_[j]/x)
            self.spacing.append(np.average(s))
        

                                
    def plane_order(self,plane):
        """
        Calculates the order of an individual plane.
        Order is defined as the average unit lattice deviation from all atoms in any direction
        """
        
        plane=np.array(plane)/np.linalg.norm(plane) #must normalize plane so miller planes can be compared.
        a=copy(self.atom_locations)
        l=copy(self.lattice_points)
        l=[[x[0]-1,x[1]-1,x[2]-1] for x in l]
        for i in range(3):
            mini=np.min(a[:,i])
            for j in range(len(a)):
                a[j][i]=(a[j][i]-mini)/self.spacing[i]# Takes atom locations, normalizes them to first atom=0
                                                      # then divides by spacing to make them integer spaces
        
        l=[np.dot(x,np.array(plane)) for x in l]
        a=[np.dot(x,np.array(plane)) for x in a]
        s=[np.abs(a[i]-x) for i,x in enumerate(l)]
    
        return np.average(s) #The average unit lattice deviation in planar direction
    
    def miller_sphere_plot(self,n=200,c1=1.2,c2=10,cross_section=False,plot=True):
        """Plotting function"""
        hkl=fibonacci_sphere(n)
        I=[self.plane_order(x) for x in hkl]
        self.I=I
        if plot:
            O=Order_plot(hkl,I,c1,c2)
        if cross_section:
            O.plot_cross_sections()
        
        

    
    
class Order_plot(): 
    
    def __init__(self, hkl,I,c1=1.2,c2=10):
        """
        Plots the data from LR_order
        Args:
            hkl: list of miller planes
            I: intensities from LR_order
            c1: controls relative peak intensities. Higher value exaggerates highest peaks more.
            c2: controls sphere size relative to peak heights. Higher value makes peaks smaller relative height.
        """
        
        #Load in Data

        h=np.array([x[0] for x in hkl])
        k=np.array([x[1] for x in hkl])
        l=np.array([x[2] for x in hkl])
        I=np.array(I)

        #Prep Intensities for density calculation
        I=np.array([x**-c1 for x in I])
        I=I/np.max(I)

        

        
        
        sigma, n =.2 , 10000
        xyzs = fibonacci_sphere(n)
        grids = np.zeros([n, 3])
        grids[:, :2] = self.xyz2sph(xyzs)
        pts = []
        for i in range(len(h)):
            p, r = self.hkl2tp(h[i], k[i], l[i])
            pts.append([p, r, I[i]])
        pts = np.array(pts)
        vals = self.calculate_density(pts, xyzs, sigma=sigma)
        
        
        #Prep heights for sphere scaling
        valss=vals+c2
        valss/=valss.max()

        for i,x in enumerate(valss):
            xyzs[i]*=np.abs(x)

   
            
        phi=[]
        rho=[]
        for row in xyzs:
            r,p=self.hkl2tp(row[0],row[1],row[2])
            phi.append(p)
            rho.append(r)
        phi=np.array(phi)
        rho=np.array(rho)
        x=xyzs[:,0]
        y=xyzs[:,1]
        z=xyzs[:,2]
        
        self.x=x
        self.y=y
        self.z=z
        
        
        points2D=np.vstack([phi,rho]).T
        tri=Delaunay(points2D)
        simplices=tri.simplices
        trisurf=create_trisurf(x=x,y=y,z=z,colormap='Jet', simplices=simplices,plot_edges=False,color_func=self.color_func,show_colorbar=True)
        self.colorscale = [
            [0, "rgb(84,48,5)"],
            [1, "rgb(84,48,5)"],
        ]
        layout = go.Layout(scene=dict(aspectmode='data',annotations=self.get_axis_names()))
        fig=go.Figure(data=trisurf, layout=layout)
        fig.add_trace(go.Scatter3d(x = [1.1,0,0], y = [0,1.1,0], z=[0,0,1.1], mode="text", text = ['a','b','c']))
        self.add_axis_arrows(fig)
        fig.update_scenes(camera_projection_type='orthographic')

        
        
        fig.show()
        
        


    
    def color_func(self,x,y,z):
        """
        Assigns color to distance
        """
        mag=np.sqrt(x**2 + y**2 + z**2)
        # return np.floor(mag*255.9999)
        return mag

    def calculate_density(self,pts, xyzs, sigma=0.1):
        """
        calculate the projected order density on the unit sphere
        uses gaussain distrbution to smooth points.
        """
        vals = np.zeros(len(xyzs))
        pi = np.pi
        for pt in pts:
            t0, p0, h = pt
            x0, y0, z0 = np.sin(t0)*np.cos(p0), np.sin(t0)*np.sin(p0), np.cos(t0)
            dst = np.linalg.norm(xyzs - np.array([x0, y0, z0]), axis=1)
            vals += h*np.exp(-(dst**2/(2.0*sigma**2)))
        return vals

    def hkl2tp(self,h, k, l):
        """
        convert hkl to theta and phi
        """
        mp = [h,k,l]
        r = np.linalg.norm(mp)

        theta = np.arctan2(mp[1],mp[0])
        phi = np.arccos(mp[2]/r)

        #return theta, phi
        return phi, theta



    def xyz2sph(self,xyzs, radian=True):
        """
        convert the vectors (x, y, z) to the sphere representation (theta, phi)

        Args:
            xyzs: 3D xyz coordinates
            radian: return in radian (otherwise degree)
        """
        pts = np.zeros([len(xyzs), 2])   
        for i, r_vec in enumerate(xyzs):
            r_mag = np.linalg.norm(r_vec)
            theta0 = np.arccos(r_vec[2]/r_mag)
            if abs((r_vec[2] / r_mag) - 1.0) < 10.**(-8.):
                theta0 = 0.0
            elif abs((r_vec[2] / r_mag) + 1.0) < 10.**(-8.):
                theta0 = np.pi

            if r_vec[0] < 0.:
                phi0 = np.pi + np.arctan(r_vec[1] / r_vec[0])
            elif 0. < r_vec[0] and r_vec[1] < 0.:
                phi0 = 2 * np.pi + np.arctan(r_vec[1] / r_vec[0])
            elif 0. < r_vec[0] and 0. <= r_vec[1]:
                phi0 = np.arctan(r_vec[1] / r_vec[0])
            elif r_vec[0] == 0. and 0. < r_vec[1]:
                phi0 = 0.5 * np.pi
            elif r_vec[0] == 0. and r_vec[1] < 0.:
                phi0 = 1.5 * np.pi
            else:
                phi0 = 0.
            pts[i, :] = [theta0, phi0]
        if not radian:
            pts = np.degree(pts)

        return pts



    def get_arrow(self,axisname="x"):
        """
        Creates arrow object to plot axis lines
        """


        body = go.Scatter3d(
            marker=dict(size=1, color=self.colorscale[0][1]),
            line=dict(color=self.colorscale[0][1], width=3),
            showlegend=False,  # hide the legend
        )

        head = go.Cone(
            sizeref=0.1,
            autocolorscale=None,
            colorscale=self.colorscale,
            showscale=False,  # disable additional colorscale for arrowheads
            hovertext=axisname,
        )
        for ax, direction in zip(("x", "y", "z"), ("u", "v", "w")):
            if ax == axisname:
                body[ax] = -1,1
                head[ax] = [1]
                head[direction] = [1]
            else:
                body[ax] = 0,0
                head[ax] = [0]
                head[direction] = [0]

        return [body, head]


    def add_axis_arrows(self,fig):
        for ax in ("x", "y", "z"):
            for item in self.get_arrow(ax):
                fig.add_trace(item)

    def get_annotation_for_ax(self,ax):
        """
        plots abc axis labels
        """
        d = dict(showarrow=False, text=ax, xanchor="left", font=dict(color="#1f1f1f"))

        if ax == "a":
            d["x"] = 1.1
            d["y"] = 0
            d["z"] = 0
        elif ax == "b":
            d["x"] = 0
            d["y"] = 1.1 
            d["z"] = 0
        else:
            d["x"] = 0
            d["y"] = 0
            d["z"] = 1.1 

        if ax in {"a", "b"}:
            d["xshift"] = 15

        return d


    def get_axis_names(self):
        return [self.get_annotation_for_ax(ax) for ax in ("a", "b", "c")]
    
    def plot_cross_sections(self):
        """
        Plots the axis plane cross sections of the plot 3D visualization
        Uses scattering of points under a limit as fibonacci sphere doesn't points distributed in a plane.
        """
        x=self.x
        y=self.y
        z=self.z
        
        e=2e-2
        xy_a=[x_ for i,x_ in enumerate(x) if np.abs(z[i])<e]
        xy_b=[z_ for i,z_ in enumerate(y) if np.abs(z[i])<e]
        xz_a=[x_ for i,x_ in enumerate(x) if np.abs(y[i])<e]
        xz_b=[z_ for i,z_ in enumerate(z) if np.abs(y[i])<e]
        yz_a=[x_ for i,x_ in enumerate(y) if np.abs(x[i])<e]
        yz_b=[z_ for i,z_ in enumerate(z) if np.abs(x[i])<e]

        fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(15,5))
        # plt.gca().set_aspect('equal', adjustable='box')
        ax[0].scatter(xy_a,xy_b,color='r')
        ax[0].set_xlim(-1,1)
        ax[0].set_ylim(-1,1)
        ax[0].set_title('X-Y')
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')
        ax[1].scatter(xz_a,xz_b,color='r')
        ax[1].set_title('X-Z')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Z')
        ax[1].set_xlim(-1,1)
        ax[1].set_ylim(-1,1)
        ax[2].scatter(yz_a,yz_b,color='r')
        ax[2].set_title('Y-Z')
        ax[2].set_xlabel('Y')
        ax[2].set_ylabel('Z')
        ax[2].set_xlim(-1,1)
        ax[2].set_ylim(-1,1)
        plt.show()
                           
def fibonacci_sphere(samples=1000):
    """
    Sampling the sphere grids

    Args:
        samples: number of pts to generate

    Returns:
        3D points array in Cartesian coordinates
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append((x, y, z))

    return np.array(points)




if __name__ == "__main__":
    

    file='S812/POSCAR.70'


    amor=amorphous_order(file,13)
    amor.miller_sphere_plot(cross_section=False)


    # fig = plt.figure()
    # ax = p3.Axes3D(fig)
    # ax.scatter(amor.atom_locations[:,0],amor.atom_locations[:,1],amor.atom_locations[:,2],color='r',s=200)
    # ax.scatter(amor.ll[:,0],amor.ll[:,1],amor.ll[:,2],color='k',s=25)
    # ax.set_xlabel('a')
    # ax.set_ylabel('b')
    # ax.set_zlabel('c')
    # plt.show()