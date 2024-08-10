import numpy as np
from tqdm import tqdm
import math

import time

def max_lattice_translation(lattice_vectors, R_max):
    # Calculate the maximum rnx, rny, and rnz
    norm_0 = np.linalg.norm(lattice_vectors[0])
    norm_1 = np.linalg.norm(lattice_vectors[1])
    norm_2 = np.linalg.norm(lattice_vectors[2])
    
    max_rnx = int(R_max / norm_0)
    max_rny = int(R_max / norm_1)
    max_rnz = int(R_max / norm_2)
    
    max_rnx, max_rny, max_rnz = np.meshgrid(np.arange(-max_rnx, max_rnx+1),
                                            np.arange(-max_rny, max_rny+1),
                                            np.arange(-max_rnz, max_rnz+1),
                                            indexing='ij')
    
    shifts = (max_rnx[..., np.newaxis] * lattice_vectors[0] + 
              max_rny[..., np.newaxis] * lattice_vectors[1] + 
              max_rnz[..., np.newaxis] * lattice_vectors[2])
    
    shift_norms = np.linalg.norm(shifts, axis=-1)
    
    valid_indices = np.where(shift_norms <= R_max)
    
    max_translation = np.array([max_rnx[valid_indices].max(),
                                max_rny[valid_indices].max(),
                                max_rnz[valid_indices].max()])
    
    return max_translation

def compute_ewald_matrix(frac_coords, lattice_vectors, sigma=None, R_max=None, G_max=None, 

                         max_shift = None,charge=None,w=1,print_info=False, triu=False):
    """
    Parameters:
    frac_coords (ndarray): Relative positions of particles (Nx3).
    lattice_vectors (ndarray): Lattice vectors of the unit cell (3x3).
    sigma (float): Ewald parameter controlling the split between real and reciprocal sums. If None, it's calculated.
    max_shift (int): Depth of the real and reciprocal space summation.
    reciprocal_depth (int): Depth of the reciprocal space summation.
    charge (ndarray): charges of the system (Nx1).

    Returns:
    ndarray: Ewald summation matrix (NxN).
    """
    from numpy.linalg import norm
    
    TO_EV = 14.39964390675221758120
    
    t0 = time.time()
    
    N = len(frac_coords)
    V = np.linalg.det(lattice_vectors)
    
    reciprocal_vectors = 2 * np.pi * np.linalg.inv(lattice_vectors).T
    
    
    # Calculate alpha if not provided
    if sigma is None:
        Sigma = ((N * w * np.pi**3) / V**2)**(-1/6)
        
    alpha = np.sqrt(1/Sigma)
    

    A = 1e-17 
    f = np.sqrt(-np.log(A))

    # Calculate R_max and G_max
    if R_max == None:
        R_max = f * np.sqrt(alpha)
        R_max = np.sqrt(-np.log(A)*Sigma**2)
    
    if G_max == None:
        G_max = 2 * f * np.sqrt(alpha)
        G_max = 2/Sigma * np.sqrt(-np.log(A))

    
    
    cart_coords = frac_coords @ lattice_vectors
    
    
    if max_shift == None:
        real_space_max = max_lattice_translation(lattice_vectors,R_max)
        reci_space_max = max_lattice_translation(reciprocal_vectors,G_max)
    
        real_reci_space_max = np.maximum(real_space_max,reci_space_max)
        
        nx = real_reci_space_max[0]
        ny = real_reci_space_max[1]
        nz = real_reci_space_max[2]
    else:
        nx = ny = nz = max_shift # To improve
    
    # Main computation
    

    
    Real_E = Reci_E = 0.
    
    Ewald_real = np.zeros((N,N))
    Ewald_recip = np.zeros((N,N))
    Ewald_recip_self = np.zeros((N,N))
    
    if print_info == True:
        print('reciprocal_vectors\n',reciprocal_vectors)

        print(f'Volume = {V}')
        print(f'w={w}')
        
        print(f'alpha= {alpha}, sigma={Sigma},R_max = {R_max}, G_max = {G_max}')
        print(f'Charge = {charge}')
        
        #print('Cart coords\n',cart_coords)
        print(f'Max vectors = {nx},{ny},{nz}')

    for i in tqdm(range(N), desc="Computing real space"):
        for j in range(i,N):
            dr = cart_coords[i]-cart_coords[j]
            dr_init = cart_coords[i]-cart_coords[j]
            dr_frac = frac_coords[i]-frac_coords[j]
            
            for rnx in range(-nx, nx + 1):
                for rny in range(-ny, ny + 1):
                    for rnz in range(-nz, nz + 1):
                        
                        lattice_translation = np.array([rnx, rny, rnz])

                        shift = rnx*lattice_vectors[0]+rny*lattice_vectors[1]+rnz*lattice_vectors[2]
                        if np.linalg.norm(shift) < R_max:
                            dr = dr_init + shift

                            if np.all(lattice_translation == 0):
                               
                                if i != j:
                                    Ewald_real[i,j] += 0.5 / norm(dr) * math.erfc(norm(dr)/Sigma) * TO_EV
                                    Real_E          += 0.5 * charge[i] * charge[j] / norm(dr) * math.erfc(norm(dr)/Sigma) * TO_EV
                                else:
                                    Ewald_recip_self[i,j] += -1 / Sigma / math.sqrt(np.pi) * TO_EV
                                    

                            else:

                                Ewald_real[i,j] += 0.5 / norm(dr) * math.erfc(norm(dr)/Sigma) * TO_EV
                                Real_E          += 0.5 * charge[i] * charge[j] / norm(dr) * math.erfc(norm(dr)/Sigma) * TO_EV
                                
                            # Reciprocal sum

                        gr = rnx*reciprocal_vectors[0]+rny*reciprocal_vectors[1]+rnz*reciprocal_vectors[2]
                        if np.linalg.norm(gr) < G_max:

                            if np.any(lattice_translation != 0):
                                g_2 = np.dot(gr, gr)

                                Reci_E += TO_EV * (2 * np.pi / V) * (charge[i] * charge[j]) * math.exp(-0.25 * Sigma * Sigma * g_2) / g_2 * math.cos(np.dot(gr, dr))
                                Ewald_recip[i,j] +=  TO_EV * (2 * np.pi / V) * math.exp(-0.25 * Sigma * Sigma * g_2) / g_2 * math.cos(np.dot(gr, dr))

    Ewald_full = Ewald_real+Ewald_recip+Ewald_recip_self
    
    for i in np.arange(N):
        for j in np.arange(i):
            Ewald_full[i, j] = Ewald_full[j, i]
            
    if triu == True:
        Ewald_tmp = np.triu(Ewald_full)*2
        np.fill_diagonal(Ewald_tmp,Ewald_full.diagonal())
        Ewald_full = np.copy(Ewald_tmp)
    
    Reci_self = sum(-(charge[i]**2) / Sigma / math.sqrt(np.pi) * TO_EV for i in range(N))
    
    if print_info == True:
        
        print(f"Real sum: {Real_E:.8f} eV")
        print(f"Reciprocal sum: {Reci_E:.8f} eV")
        print(f"Reciprocal self (eV): {Reci_self:.16f}")
        print(f"Reciprocal (eV): {Reci_self + Reci_E:.16f}")
        print(f"Total (eV): {Real_E + Reci_E + Reci_self:.16f}")
    
    return Ewald_full


def calculate_ewald_matrix_charges(ewald_matrix, charges):

    """
    Calculate the potential energy of the system given the Ewald summation matrix and charges.

    Parameters:
    ewald_matrix (ndarray): Ewald summation matrix (NxN).
    charges (ndarray): Charges of the particles (N).

    Returns:
    float: Total potential energy of the system.
    """
    
    charges = np.array(charges)
    
    return  charges[:, np.newaxis] * charges[np.newaxis, :] * ewald_matrix


def buckingham_potential(param, r):
    """
    Calculate the Buckingham potential for a given distance.

    Parameters:
    A (float): Constant A in the Buckingham potential equation.
    Rho (float): Constant Rho in the Buckingham potential equation.
    C (float): Constant C in the Buckingham potential equation.
    r (float): Distance between two atoms.

    Returns:
    float: Potential energy at distance r.
    """
    A = param[0]
    Rho = param[1]
    C = param[2]
    
    V = A * np.exp(-r / Rho) - C / r**6
    return V


def compute_buckingham_matrix(structure, buckingham_dict, R_max, max_shift=None):
    """
    Compute the Ewald summation matrix for a system of particles.

    Parameters:
    positions (ndarray): Relative positions of particles (Nx3).
    lattice_vectors (ndarray): Lattice vectors of the unit cell (3x3).

    """
    frac_coords = structure.frac_coords
    lattice_vectors = structure.lattice.matrix
    sites = structure.sites
    
    TO_EV = 14.39964390675221758120
    
    t0 = time.time()
    
    N = structure.num_sites
    V = np.linalg.det(lattice_vectors)
    
    cart_coords = frac_coords @ lattice_vectors
    
    if max_shift == None:
        max_real = max_lattice_translation(lattice_vectors, R_max)
        nx = max_real[0]
        ny = max_real[1]
        nz = max_real[2]
    else:
        ny = nz = nx = max_shift
    #print(f'Max vectors = {nx},{ny},{nz}')
    buckingham_matrix = np.zeros((N,N))
    
    for i in tqdm(range(N), desc="Buckingham matrix"):
        for j in range(i+1,N):
            
            sites_label = f'{sites[i].specie}-{sites[j].specie}'
            if sites_label in buckingham_dict:
                dr_init = cart_coords[i]-cart_coords[j]

                for rnx in range(-nx, nx + 1):
                    for rny in range(-ny, ny + 1):
                        for rnz in range(-nz, nz + 1):

                            lattice_translation = np.array([rnx, rny, rnz])

                            shift = rnx*lattice_vectors[0]+rny*lattice_vectors[1]+rnz*lattice_vectors[2]
                            dr = dr_init + shift
                            if np.linalg.norm(dr) < R_max:
                                dr = dr_init + shift
                                                                        np.linalg.norm(dr)))
                                buckingham_matrix[i][j] += buckingham_potential(buckingham_dict[sites_label],
                                                                         np.linalg.norm(dr))
    return buckingham_matrix


def build_qubo_from_Ewald_IP(Ewald_matrix,IP_matrix):
    
    return Ewald_matrix + IP_matrix