
import numpy as np
from ase.io import read, write
from ase.build.supercells import make_supercell
import matplotlib.pyplot as plt
from copy import deepcopy
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







class PyLRO:
    
    def __init__(self,filename,atom_number,fileformat='vasp'):
        """
        Args:
            filename: a crystal structure file
            atom_number: the atomic number of the element chosen for miller plane analysis. Only supports unit n=1 in unit cell
            fileformat: ASE accepted file format
        
        Returns:
            plotly 3dlpot of the miller sphere scaled to relative periodicities.
        """
        
        
        
        self.struc = read(filename,format=fileformat)
        self.atom_number=atom_number
        original = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.struc = make_supercell(self.struc, original) 
        if atom_number==None: #lowest atomic number species by default to calculate periodicity
            unique, counts = np.unique(self.struc.numbers, return_counts=True)
            atom_number=unique[np.argmin(counts)]
        self.cell=self.struc.cell
        idx=[i for i,x in enumerate(self.struc.numbers) if x==atom_number]
        points_=self.struc.get_scaled_positions() #grab scaled, <abc> independent positions
        self.atom_locations=np.array([points_[i] for i in idx])
        

                


        

    def lattice_fit(self,n=13):
        """
        Creates a best fit lattice for the structure. Uses structure factor to determine periodicity among a,b,c directions.
        Args:
            n: the size limit of the sample supercell of lattice points
        Returns:
            dimensions: 1x3 array, size of cell
            lattice_repr: nx3 array, integer representation of atomic locations
            
        """
        

        def structure_factor(pos, hkl):
            """ N*1 array"""
            F = 0
            h, k, l = hkl
            for xyz in pos:
                x,y,z = xyz
                F += np.exp(-2*np.pi*(1j)*(h*x + k*y+ l*z))

            return F
        
        
        def fit(n,al_,test=False):#fix periodic boundary
            al_=[x*n for x in al_]
            al=deepcopy(np.array(al_))
            v=al
            if np.average([np.abs(np.round(x)-x) for x in v])<.1:
                v=[int(np.round(x)) for x in v]
                if np.any([x<.1 for x in v]):
                    v=[x+1 for x in v]
                return v
            
            v_mod=[x%1 for x in v]
            intshift=np.ones(len(v_mod))
            for i,v_ in enumerate(v_mod):
                if v_-.5>0:
                    v_mod[i]=v_-1
                    intshift[i]=0
            v_avg=np.average(v_mod)
            
            
            v=[int(np.round(x-v_avg)) for i,x in enumerate(v)]
            if 0 in v:
                v=[x+1 for x in v]
            return v

        al=deepcopy(self.atom_locations)
        renorm=[False,False,False]
        for i in range(3):
            high=np.max(al[:,i])
            low=np.min(al[:,i])
            if np.abs((1-high)-low)<.1:
                renorm[i]=True


        cdim=[]
        sfac=[]
        lattice_repr=[]
        basis=np.array([[1,0,0],[0,1,0],[0,0,1]])
        for j,b in enumerate(basis):
                
            ss=[np.abs(structure_factor(al,b*x)) for x in range(1,n)]
            ss_idx=np.argsort(-np.array(ss))
            continuous=False
            c=0
            while not continuous:
                dim=ss_idx[c]+1
                points=fit(dim,al[:,j])

                if len(np.unique(points))<dim:
                    c+=1
                else:
                    if renorm[j]:
                        dim=dim+1
                        self.atom_locations[:,j]=self.atom_locations[:,j]*((dim-1)/dim)
                        points=fit(dim,self.atom_locations[:,j],test=True)
                    cdim.append(dim)
                    sfac.append(ss[ss_idx[c]])
                    lattice_repr.append(points)
                    continuous=True
        self.dimensions=cdim
        self.lattice_repr=np.transpose(lattice_repr)

        

        self.avgfac=np.average(sfac)
        

        a=deepcopy(self.atom_locations)
        dim=deepcopy(self.lattice_repr)
        m=len(a[:,0])
        
        lis=np.zeros([m,3])

        for i in range(3):
            base=np.min(a[:,i])
            for j in range(m):
                a[j,i]=a[j,i]-base
                lis[j,i]=dim[j,i]-1.
            maxx=np.max(a[:,i])
            max_=np.max(lis[:,i])
            for j in range(m):
                lis[j,i]=lis[j,i]/max_*maxx
        

        self.x,self.y,self.z=np.transpose(a)
        self.x_,self.y_,self.z_=np.transpose(lis)
        

            

    

    
    def calculate_spacing(self):
        """
        Once the atoms are assigned to a lattice, the average scaled distance between calculated.
        Function normalizes the first points, then calculates average scaled spacing in a,b,c directions
        The spacing needs to be normalized so that planes in non basis directions can have relative deviations combined
        """
        a=deepcopy(self.atom_locations)
        l=deepcopy(self.lattice_points)
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
        

                                
    def plane_order(self,plane,angstrom=False):
        """
        Calculates the order of an individual plane.
        Order is defined as the average unit lattice deviation from all atoms in any direction
        """
        
        plane=np.array(plane)/np.linalg.norm(plane) #must normalize plane so miller planes can be compared.
        a=deepcopy(self.atom_locations)
        l=deepcopy(self.lattice_repr)-1
        n1,n2,n3=self.dimensions
        for i,x in enumerate(a):
            a[i]=[x[0]*n1,x[1]*n2,x[2]*n3]
        for i in range(3): #fix periodic boundary
            a_=a[:,i]
            amods=[x%1 for x in a_]
            for j,aa in enumerate(amods):
                if aa-.5>0:
                    amods[j]=aa-1.
            avgmod=np.average(amods)
            a_=[x-avgmod for x in a_]
            a[:,i]=a_
            if np.min(a[:,i])-.5>0:
                a[:,i]-=1
            
        s=[np.abs(a[i]-x) for i,x in enumerate(l)]
        s_=[np.dot(x,self.cell) for x in s]
        self.absolute_disorder=[np.abs(np.dot(x,np.array(plane))) for x in s_]
        self.relative_disorder=[np.abs(np.dot(x,np.array(plane))) for x in s]
        
        if angstrom:
            return np.average(self.absolute_disorder)
        return np.average(self.relative_disorder)
    
        
    def maximum_order(self,n=700):
        hkl=fibonacci_sphere(n)
        I=[self.plane_order(x) for x in hkl]
        mags=[np.linalg.norm(x) for x in I]
        return (np.min(mags), hkl[np.argmin(mags)])
    
    def minimum_order(self,n=700):
        hkl=fibonacci_sphere(n)
        I=[self.plane_order(x) for x in hkl]
        mags=[np.linalg.norm(x) for x in I]
        return (np.max(mags), hkl[np.argmax(mags)])
        
    
    def miller_sphere_plot(self,n=700,c1=1.2,c2=10,cross_section=False,plot=True,angstrom=False):
        """Plotting function"""
        hkl=fibonacci_sphere(n)
        if not angstrom:
            I=[self.plane_order(x) for x in hkl]
        if angstrom:
            I=[self.plane_order(x,angstrom=True) for x in hkl]

            
        

        self.I=I
        if plot:
            O=Order_plot(hkl,I,c1,c2,angstrom=angstrom)
        if cross_section:
            O.plot_cross_sections()
        
        

    
    
class Order_plot(): 
    
    def __init__(self, hkl,I,c1=1.2,c2=10,angstrom=False):
        """
        Plots the data from LR_order
        Args:
            hkl: list of miller planes
            I: intensities from LR_order
            c1: controls relative peak intensities. Higher value exaggerates highest peaks more.
            c2: controls sphere size relative to peak heights. Higher value makes peaks smaller relative height.
        """
        
        #Load in Data
        self.hkl=hkl
        h=np.array([x[0] for x in hkl])
        k=np.array([x[1] for x in hkl])
        l=np.array([x[2] for x in hkl])
        I=np.array(I)

        #Prep Intensities for density calculation
        # I=np.array([x**-c1 for x in I])
        if angstrom:
            mmm=np.max(I)
        else:
            I=np.array([1-x for x in I])
            mmm=np.max(I)
 
        self.I=I
        # I=I/np.max(I)

        

        
        
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
        valss=vals
        valss/=valss.max()
        valss*=mmm


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
        cmap=self.colormap_gen_(np.min(I),np.max(I))
        if angstrom:
            cmap='Jet'
        
        
        
        points2D=np.vstack([phi,rho]).T
        tri=Delaunay(points2D)
        simplices=tri.simplices

        trisurf=create_trisurf(x=x,y=y,z=z,colormap=cmap, simplices=simplices,plot_edges=False,color_func=self.color_func_,show_colorbar=True)
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
        
        


    
    def colormap_gen_(self,low,high,threshold=.8):
        blue=np.array([0,0,1])
        yellow=np.array([1,1,0])
        red=np.array([1,0,0])
        mid=1-(1-threshold)/2
        if low<mid and high <=mid:
            x1=(low-threshold)/(mid-threshold)
            if x1<0:
                x1=0
            y1=1-x1
            x2=(high-threshold)/(mid-threshold)
            if x2<0:
                x2=0
            y2=1-x2
            c1=x1*yellow+y1*blue
            c2=x2*yellow+y2*blue
            c1[c1>1]=1
            c2[c2>1]=1
            
            return [tuple(c1),tuple(c2)]
            
            
        if low>mid and high>=mid:
            x1=(low-mid)/(1-mid)
            y1=1-x1
            x2=(high-mid)/(1-mid)
            y2=1-x2
            c1=x1*red+y1*yellow
            c2=x2*red+y2*yellow
            c1[c1>1]=1
            c2[c2>1]=1
            return [tuple(c1),tuple(c2)]
            
        if low<mid and high>mid:
            x1=(low-threshold)/(mid-threshold)
            if x1<0:
                x1=0
            y1=1-x1
            x2=(high-mid)/(1-mid)
            y2=1-x2
            c1=x1*yellow+y1*blue
            c2=x2*red+y2*yellow
            center=(low+high)/2
            if center>mid:
                x_=(center-mid)/(1-mid)
                y_=1-x_
                c_=x_*yellow+y_*red
                return [tuple(c1),tuple(c_),tuple(c2)]
            else:
                x_=(center-threshold)/(mid-threshold)
                y_=1-x_
                c_=x_*blue+y_*yellow
                return [tuple(c1),(1,1,0),tuple(c2)]
            
            
            
            
        
        

        
    def color_func(self,x,y,z):
        """
        Assigns color to distance
        """
        arr=np.array([x,y,z])
        arr_=[np.linalg.norm(arr-np.array(x)) for x in self.hkl]
        mag=self.I[np.argmax(arr_)]
        # mag=np.sqrt(x**2 + y**2 + z**2)
        # return np.floor(mag*255.9999)
        return mag
    def color_func_(self,x,y,z):
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

