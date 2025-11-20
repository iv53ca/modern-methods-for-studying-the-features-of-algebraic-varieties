import numpy as np
from scipy.spatial import ConvexHull,convex_hull_plot_2d
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols, Function, Eq, solve, I, collect, expand, simplify,\
                  Derivative, init_printing, series, evaluate, Poly, Rational,\
                  gcd, eye, Matrix, sign, ln, exp

from .pgutils import Norm2Dlst


def NPimage(CH,S,ecol="k",vcol="b",withNormals=False,savefig=None):
    """
     Returns plot object of Newton polygon 
     Parameters ecol and vcol define colors of edges and vertices
     If withNormals is True the external normals are drawn
     If savefig is a string the image is saved with the given name
     and extension "png"
    """
    def getlimits(S):
        npS = np.array(S,dtype=np.int32)
        return np.array([np.min(npS[:,0]),np.max(npS[:,0]),np.min(npS[:,1]),\
                         np.max(npS[:,1])],dtype=np.int32)
    delta = 0.4
    polylimits = getlimits(S)
    enlarge = np.array([-delta,delta,-delta,delta],dtype=np.float64)
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    ax.axis('square')
    ax.axis(polylimits+enlarge)
    majorx_ticks = np.arange(polylimits[0],polylimits[1]+1,1)
    majory_ticks = np.arange(polylimits[2],polylimits[3]+1,1)
    ax.set_xticks(majorx_ticks)
    ax.set_yticks(majory_ticks)
    ax.grid(which='major',color="grey",linestyle='-',lw=1,alpha=0.5)
    ax.set_xlabel("$q_1$",fontsize=16)
    ax.set_ylabel("$q_2$",fontsize=16,rotation=0)
    if withNormals:
        CH_NC = Norm2Dlst(CH)
    for i,edge in enumerate(CH.simplices):
        ax.plot(S[edge,0], S[edge,1],ecol+'-', lw=2)
        if withNormals:
            origin=(S[edge][0,:]+S[edge][1,:])/2
            plt.quiver(*origin,[CH_NC[i][0]],[CH_NC[i][1]],color=['b'],scale=10)
    ax.plot(S[:,0], S[:,1], vcol+'o')
    if isinstance(savefig,str): 
        plt.savefig(savefig+".png",dpi=300,bbox_inches='tight')
    #plt.show()
    return ax

def AddEdgeLabel(ax,S,CH,normlst,edgenum,text,shift=0.25):
    """
    Put given text near the edge with number edgenum
    Position of the text is selected with normlst and shifted out with shift
    """
    midpnt = (S[CH.simplices[edgenum][0]]+S[CH.simplices[edgenum][1]])/2
    lblpos = midpnt+shift*normlst[edgenum][:-1]
    ax.text(*lblpos,text,fontsize=16)#,position=lblpos)