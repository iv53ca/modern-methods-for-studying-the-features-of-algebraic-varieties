import numpy as np
from scipy.spatial import ConvexHull,convex_hull_plot_2d
import sympy as sym
from sympy import symbols, Function, Eq, solve, I, collect, expand, simplify,\
                  Derivative, init_printing, series, evaluate, Poly, Rational,\
                  gcd, eye, Matrix, sign, ln, exp, lcm
from functools import reduce
from copy import copy,deepcopy
from itertools import chain
                  
x,y = symbols("x y",real=True)

def Rational2integer(rational_vector):
    """
    Convert a vector of rational numbers (Sympy Rational objects) into a 
    collinear integer vector in numpy array format.

    Parameters:
    rational_vector (list): A list of sympy Rational numbers representing the vector.

    Returns:
    np.array: A numpy array of integers that is collinear to the input vector.
    """
    # Step 1: Extract the denominators of each rational component in the vector
    denominators = [rational.q for rational in rational_vector]
    # Step 2: Calculate the least common multiple (LCM) of all denominators
    common_denominator = reduce(lcm,denominators)
    # Step 3: Scale each component by the LCM to make all components integers
    integer_vector = [int(rational * common_denominator) for rational in rational_vector]
    # Step 4: Convert the integer vector to a numpy array
    collinear_integer_vector = np.array(integer_vector,dtype=np.int32)
    
    return collinear_integer_vector

def MkSPSubs(SP:dict,newvars:[list|tuple])->dict:
    """ 
    Compute substitution old variables based on the singular point coordinates 
    in the form of dictionary {oldvar:value}
    """
    if len(SP.keys())!=len(newvars): return None
    return dict((old,SP[old]+new) for old,new in zip(SP.keys(),newvars))

def SCH(f,varlst=(x,y)):
    """
    Compute support and its convex hull. If f is not a Poly then convert it
    """
    if not isinstance(f,Poly): fpoly = Poly(f,varlst)
    else: fpoly = f
    #support=np.array([[x,y] for x,y in fpoly.monoms()],dtype=np.int32)
    support=np.array([p for p in fpoly.as_dict().keys()],dtype=np.int32)
    CH=ConvexHull(support)
    #print(support[CH.vertices])
    return support,CH
    
def Norm3Dlst(CH):
    """
    Compute the list of normals to all the faces of the convex hull CH
    """
    eqnlst = CH.equations
    normlst = []
    for eq in eqnlst:
        nvec = copy(eq[:-1])
        abs_min = np.min(np.abs(nvec[np.nonzero(nvec)]))
        nvec /= abs_min
        frac = np.array([Rational(v).limit_denominator(100) for v in nvec])
        #denom_lst = np.array([f.denominator for f in frac],dtype=np.int32)
        #lcm = np.lcm.reduce(denom_lst)
        #normlst.append((frac*lcm).tolist())
        normlst.append(Rational2integer(frac))
    return normlst

def Norm2Dlst(CH):
    """
    Returns the list of polygon's normals
    """
    eps = 1e-7
    edgelst = CH.simplices
    eqnlst = CH.equations
    #midpntlst = np.array([(S[edge[0]]+S[edge[1]])/2 for edge in edgelst],dtype=np.float64)
    normlst = []
    for eq in eqnlst:
        if np.abs(eq[0]*eq[1])>eps:
            frac = Rational(*eq[:-1]).limit_denominator(100)
            normal = int(np.sign(eq[1]))*np.array([frac.numerator,frac.denominator],dtype=np.int32)
        #normal = normal + [eq[-1]*frac.denominator/np.abs(eq[1])]
            normlst.append(normal)
        else: 
            normal = np.array(eq[:-1],dtype=np.int32)#+[eq[-1]]
            normlst.append(normal)
    #normlst = np.array(normlst,dtype=np.int32)
    return normlst #list(zip(normlst,midpntlst))

    
def GetTrunc(f,CH,edgenum,varlst=(x,y),factorize=True):
    """
    Return truncated equation corresponeding to the edge with number edgenum
    """
    eps = 1e-7
    if not isinstance(f,Poly): fpoly = Poly(f,varlst)
    else: fpoly = f
    if edgenum >= len(CH.equations): return Null
    fdict = fpoly.as_dict()
    trunc = Poly.from_dict({p:fdict[p] for p in fdict.keys()\
                            if np.abs(np.dot(np.append(p,1),CH.equations[edgenum]))<eps},\
                           *varlst).as_expr()
    if factorize: return trunc.factor()
    else: return trunc