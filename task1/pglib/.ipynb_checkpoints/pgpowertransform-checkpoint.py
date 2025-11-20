import numpy as np
import sympy as sym
from sympy import symbols, Function, Eq, solve, I, collect, expand, simplify,Derivative, init_printing, series, evaluate, Poly,\
                Rational,gcd, eye, Matrix, sign, ln, exp
from sympy.ntheory.continued_fraction import continued_fraction_convergents,\
    continued_fraction_iterator, continued_fraction
                  
def UniMod1(p,q):
    """
    Construct 2x2 unimodular matrix for fraction p/q.
    First variant
    """
    pabs=abs(p)
    qabs=abs(q)
    gcdpq=gcd(pabs,qabs)
    if gcdpq!=1 :
        pabs//=gcdpq
        qabs//=gcdpq
    r = Rational(pabs,qabs)
    cfr = continued_fraction(r)
    Mlst = [eye(2) for k in range(len(cfr))]
    for k,m in enumerate(cfr):
        if k%2==1: Mlst[k][1,0]=-m
        else: Mlst[k][0,1]=-m
    alpha = eye(2)
    for M in Mlst[::-1]: alpha*=M
    alpha[:,0]*=sign(p)
    alpha[:,1]*=sign(q)
    return alpha

def UniMod2(p,q):
    """
    Construct 2x2 unimodular matrix for fraction p/q.
    Second variant
    """
    pabs=abs(p)
    qabs=abs(q)
    gcdpq=gcd(pabs,qabs)
    if gcdpq!=1 :
        pabs//=gcdpq
        qabs//=gcdpq
    
    r = Rational(pabs,qabs)
    cfr = continued_fraction(r)
    cfr_conv = list(continued_fraction_convergents(cfr))
    gamma = cfr_conv[-1]
    rho = cfr_conv[-2]
    sigma=(gamma-rho).numerator
    alpha = Matrix([[sign(p)*sigma*rho.denominator,-sign(q)*sigma*rho.numerator],[-sign(p)*gamma.denominator,sign(q)*gamma.numerator]])
    return alpha

def make_permute(A):
    """
    Sorts a list of integers A by absolute value in descending order.
    Returns the sorted array and the permutation matrix.
    """
    if isinstance(A,list): Avec = np.array(A)
    else: Avec = A
    A_indices = np.argsort(-np.abs(Avec))
    Asort = Avec[A_indices]
    
    # Create a permutation matrix based on the sorting
    nA = len(A)
    Aper = np.zeros((nA, nA), dtype=int)
    for i, index in enumerate(A_indices):
        Aper[i, index] = 1
        
    return Asort, Aper

def make_unimod(As):
    """
    Computes a unimodular matrix to transform a sorted vector into a new form 
    with zeros in all but one position. Adjusts for positive or negative 
    elements to maintain a single non-zero element in the last position.
    """
    n = len(As)
    M = np.eye(n, dtype=int)
    
    abs_As = np.abs(As)
    
    # Find the minimal element's value and its last occurrence
    nminpos = np.searchsorted(-abs_As,0)
    # Set the column index for operations based on the last occurrence of the minimal element
    ncol = nminpos - 1
    
    # Populate the unimodular matrix to zero out elements above the last minimal element
    for i in range(ncol):
        if As[ncol] != 0:
            multiplier = -(np.abs(As[i]) // np.abs(As[ncol])*np.sign(As[i])*np.sign(As[ncol]))
            M[i, ncol] = multiplier
    
    return M

def Unimod_recursive(A, Uni=None):
    """
    Recursively applies Unimod transformation until A has only one non-zero component
    and ensures that this non-zero component is at the last position.
    """
    Alen = len(A)
    if Uni is None:
        res = np.eye(Alen, dtype=int)
    else:
        res = Uni
    
    # Base case: If only one non-zero element, ensure it is at the last position
    if np.count_nonzero(A) == 1:
        non_zero_index = np.nonzero(A)[0][0]
        if non_zero_index != Alen - 1:
            final_perm = np.eye(Alen, dtype=int)
            final_perm[[non_zero_index, Alen - 1]] = final_perm[[Alen - 1, non_zero_index]]
            res = final_perm @ res
            A = final_perm @ np.array(A)
        return A, res

    As, Aper = make_permute(A)
    M = make_unimod(As)
    
    Auni = M @ Aper
    transformed_A = Auni @ np.array(A)

    return Unimod_recursive(transformed_A, Auni @ res)

def PowTrans(alpha,oldvars,newvars):
    """
    Define power transformation according to matrix alpha
    """
    if len(oldvars)!=len(newvars):
        raise ValueError
    lnnewvars=Matrix([list(map(sym.ln,newvars))]).T
    return dict([(oldvar,newvar) for oldvar,newvar in zip(oldvars,list(map(sym.exp,alpha@lnnewvars)))])