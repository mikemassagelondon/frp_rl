import jax.numpy as jnp
from jax import grad, jit
import math
import numpy as np

def empirical_effective_dim_ratio(singular_values, r):
    """
    Calculate the normalized effective dimension for given singular values and r.
    
    Parameters:
        singular_values (numpy.ndarray): Singular values of a matrix
        r (float): Regularization parameter in (0, 1)
        
    Returns:
        float: Normalized effective dimension
    """
    s_squared = singular_values ** 2
    return np.mean(s_squared / (s_squared + r))

def create_F_function(r, beta, ell, with_mp=False, c=1.):
    """
    Create the F function for Newton's method.
    Solve   -1/r = y (βy+1)^ℓ / (y+1)^{ℓ+1},  y ∈ (-1,0)
    RHS = chi(y) =  S(y)y/(1+y),  
    where S is the S-transform of free product of 
        (u_1+\dots + u_n)^*(u_1+\dots +u_n)/n,
    given by 
        s(z)= [(z/n + 1)/(z+1) ]^ell.
    If with_mp=True, we futher multiply with MP_law (param c): 
        S(z)= z/(z+c)
     
    Parameters:
        r (float): Regularization parameter
        beta (float): Beta parameter (for treating 1/n)
        ell (int): Ell parameter
        with_mp (bool): product with Marchenko Pastur (=free poison law)
        c (float): MP parameter
    Returns:
        function: 
            if with_mp = false
                F(y) = ry(βy+1)^ℓ + (y+1)^{ℓ+1}
            otherwise
                F(y) = ry(βy+1)^ℓ + (y+1)^{ℓ+1}(y+c)
            (with applying y=-y)
    """
    if with_mp:
        def F(y):
            y = -y # since \psi = - d_eff/d
            return r*y * (beta * y + 1.0)**ell + (y+c)*(y + 1.0)**(ell + 1)
    else:
        def F(y):
            y = -y 
            return r*y * (beta * y + 1.0)**ell + (y + 1.0)**(ell + 1)
    
    return F

def newton_method(F, y0=0.5, tol=1e-7, max_iter=1000):
    """
    Solve F(y)=0 (0<y<1) using Newton's method.
    
    Parameters:
        F: target function
        y0 (float, optional): Initial guess for y. Defaults to 0.5.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-7.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        
    Returns:
        tuple: (root, iterations) - The solution and number of iterations taken
    """
    Fprime = jit(grad(F))  # automatic derivative
    
    @jit
    def newton_step(y):
        return y - F(y) / Fprime(y)
    
    y = y0
    for k in range(max_iter):
        y_next = newton_step(y)
        # clamp to (-1,0) to respect the domain mathematically
        y_next = jnp.clip(y_next, 1e-16, 0.999999999999 )
        if jnp.abs(F(y_next)) < tol:
            return float(y_next), k + 1
        y = y_next
    
    raise RuntimeError(f"Newton did not converge in {max_iter} iterations")

def free_effective_dim_ratio(r, beta, ell, with_mp=False, c=1, y0=0.5, tol=1e-7, max_iter=1000):
    """
    Calculate the effective dimension using the theoretical formula.
    
    Parameters:
        r (float): Regularization parameter
        beta (float): Beta parameter
        ell (int): Ell parameter
        y0 (float, optional): Initial guess for y. Defaults to 0.5.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-7.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        
    Returns:
        float: The effective dimension
    """
    F = create_F_function(r, beta, ell, with_mp=with_mp, c=c)
    root, _ = newton_method(F, y0, tol, max_iter)
    #u = -1/r
    #psi = -root  # \psi(u)
    #G = u * (1 + psi)  # G(1/u)=G(-r)
    #effective_dim = 1 + r * G  # \tau(K(K+r)^{-1}) = root
    return root

if __name__ == "__main__":
    # Example usage when run as a script
    logn = 8
    ell = 1
    k = round(logn/2)
    beta = math.pow(2, -k)  # β ∈ (0,1]
    r = 0.001  # r > 0, for tr(K(K+r)^{-1})
    with_mp=True
    c=1
    
    # Create the F function first
    F = create_F_function(r, beta, ell, with_mp=with_mp,c=c)
    
    # Calculate root using Newton's method with increased max_iter and adjusted initial guess
    root, iters = newton_method(F, y0=0.5, tol=1e-7, max_iter=1000)
    
    # Print results
    print(f"β={beta}, ℓ={ell}, r={r}")
    print(f"root y ≈ {root:.16g}  (converged in {iters} iterations)")
    y=-root
    if with_mp:
        print(f"check:  LHS = {-1/r:.16g},  RHS = {y*(beta*y+1)**ell/((y+c)*(y+1)**(ell+1)):.16g}")
    
    else:
        print(f"check:  LHS = {-1/r:.16g},  RHS = {y*(beta*y+1)**ell/(y+1)**(ell+1):.16g}")
    
    # Calculate effective dimension
    effective_dim = free_effective_dim_ratio(r, beta, ell,with_mp=with_mp, c=c)
    print(f"effective dimension: {effective_dim}")
