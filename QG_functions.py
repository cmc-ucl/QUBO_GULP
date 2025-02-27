import numpy as np
from tqdm import tqdm
import math
from numba import njit, prange

import time

def max_lattice_translation_old(lattice_vectors, R_max):
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

                         max_shift = None,w=1,print_info=False, triu=False):
    """
    Parameters:
    frac_coords (ndarray): Relative positions of particles (Nx3).
    lattice_vectors (ndarray): Lattice vectors of the unit cell (3x3).
    sigma (float): Ewald parameter controlling the split between real and reciprocal sums. If None, it's calculated.
    max_shift (int): Depth of the real and reciprocal space summation.
    reciprocal_depth (int): Depth of the reciprocal space summation.

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
        real_space_max = max_lattice_translation_old(lattice_vectors,R_max)
        reci_space_max = max_lattice_translation_old(reciprocal_vectors,G_max)
    
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
                                else:
                                    Ewald_recip_self[i,j] += -1 / Sigma / math.sqrt(np.pi) * TO_EV
                                    

                            else:

                                Ewald_real[i,j] += 0.5 / norm(dr) * math.erfc(norm(dr)/Sigma) * TO_EV
                                
                                
                            # Reciprocal sum

                        gr = rnx*reciprocal_vectors[0]+rny*reciprocal_vectors[1]+rnz*reciprocal_vectors[2]
                        if np.linalg.norm(gr) < G_max:

                            if np.any(lattice_translation != 0):
                                g_2 = np.dot(gr, gr)

                                Ewald_recip[i,j] +=  TO_EV * (2 * np.pi / V) * math.exp(-0.25 * Sigma * Sigma * g_2) / g_2 * math.cos(np.dot(gr, dr))

    Ewald_full = Ewald_real+Ewald_recip+Ewald_recip_self
    
    for i in np.arange(N):
        for j in np.arange(i):
            Ewald_full[i, j] = Ewald_full[j, i]
            
    if triu == True:
        Ewald_tmp = np.triu(Ewald_full)*2
        np.fill_diagonal(Ewald_tmp,Ewald_full.diagonal())
        Ewald_full = np.copy(Ewald_tmp)
    
    
    
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
                                buckingham_matrix[i][j] += buckingham_potential(buckingham_dict[sites_label],
                                                                         np.linalg.norm(dr))
    return buckingham_matrix


def build_qubo_from_Ewald_IP(Ewald_matrix,IP_matrix):
    
    return Ewald_matrix + IP_matrix

### NUMBA FUNCTIONS

# Keep this function outside Numba
def max_lattice_translation(lattice_vectors, R_max):
    norm_0 = np.linalg.norm(lattice_vectors[0])
    norm_1 = np.linalg.norm(lattice_vectors[1])
    norm_2 = np.linalg.norm(lattice_vectors[2])

    max_rnx = int(R_max / norm_0)
    max_rny = int(R_max / norm_1)
    max_rnz = int(R_max / norm_2)

    return np.array([max_rnx, max_rny, max_rnz])


# Use Numba to optimize the main function
@njit(parallel=True)
def compute_ewald_matrix_numba(frac_coords, cart_coords, reciprocal_vectors, sigma, R_max, 
                               G_max, nx, ny, nz, TO_EV, Sigma, alpha, V, N):
    
    Ewald_real = np.zeros((N, N))
    Ewald_recip = np.zeros((N, N))
    Ewald_recip_self = np.zeros((N, N))

    for i in prange(N):
        for j in prange(i, N):
            dr_init = cart_coords[i] - cart_coords[j]

            for rnx in prange(-nx, nx + 1):
                for rny in prange(-ny, ny + 1):
                    for rnz in prange(-nz, nz + 1):
                        lattice_translation = np.array([rnx, rny, rnz])
                        shift = rnx * lattice_vectors[0] + rny * lattice_vectors[1] + rnz * lattice_vectors[2]

                        if np.linalg.norm(shift) < R_max:
                            dr = dr_init + shift

                            if np.all(lattice_translation == 0):
                                if i != j:
                                    Ewald_real[i, j] += 0.5 / np.linalg.norm(dr) * math.erfc(np.linalg.norm(dr) / Sigma) * TO_EV
                                else:
                                    Ewald_recip_self[i, j] += -1 / Sigma / math.sqrt(np.pi) * TO_EV
                            else:
                                Ewald_real[i, j] += 0.5 / np.linalg.norm(dr) * math.erfc(np.linalg.norm(dr) / Sigma) * TO_EV

                        gr = rnx * reciprocal_vectors[0] + rny * reciprocal_vectors[1] + rnz * reciprocal_vectors[2]
                        if np.linalg.norm(gr) < G_max:
                            if np.any(lattice_translation != 0):
                                g_2 = np.dot(gr, gr)
                                Ewald_recip[i, j] += TO_EV * (2 * np.pi / V) * math.exp(-0.25 * Sigma * Sigma * g_2) / g_2 * math.cos(np.dot(gr, dr_init))

    Ewald_full = Ewald_real + Ewald_recip + Ewald_recip_self

    for i in prange(N):
        for j in prange(i):
            Ewald_full[i, j] = Ewald_full[j, i]

    return Ewald_full

# Call the function
def main_ewald_computation(frac_coords, lattice_vectors, sigma=None, R_max=None, G_max=None, max_shift=None, w=1, print_info=False, triu=False):
    TO_EV = 14.39964390675221758120
    
    N = len(frac_coords)
    V = np.linalg.det(lattice_vectors)
    
    reciprocal_vectors = 2 * np.pi * np.linalg.inv(lattice_vectors).T

    # Calculate alpha if not provided
    if sigma is None:
        Sigma = ((N * w * np.pi**3) / V**2)**(-1/6)
        
    alpha = np.sqrt(1 / Sigma)
    A = 1e-17
    f = np.sqrt(-np.log(A))

    # Calculate R_max and G_max
    if R_max is None:
        R_max = np.sqrt(-np.log(A) * Sigma**2)
    
    if G_max is None:
        G_max = 2 / Sigma * np.sqrt(-np.log(A))

    cart_coords = frac_coords @ lattice_vectors

    # Precompute max lattice translation values outside Numba
    if max_shift is None:
        real_space_max = max_lattice_translation(lattice_vectors, R_max)
        reci_space_max = max_lattice_translation(reciprocal_vectors, G_max)
        real_reci_space_max = np.maximum(real_space_max, reci_space_max)
        nx, ny, nz = real_reci_space_max
    else:
        nx = ny = nz = max_shift

    # Call the Numba-optimized function
    Ewald_full = compute_ewald_matrix_numba(frac_coords, cart_coords, reciprocal_vectors, sigma, R_max, G_max, nx, ny, nz, TO_EV, Sigma, alpha, V, N)

    # Symmetrize the matrix
    if triu:
        Ewald_tmp = np.triu(Ewald_full) * 2
        np.fill_diagonal(Ewald_tmp, Ewald_full.diagonal())
        Ewald_full = np.copy(Ewald_tmp)

    return Ewald_full


def calculate_ewald_charges(ewald_matrix, charges):

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

# Numba-friendly Buckingham potential function
@njit
def buckingham_potential(buckingham_params, r):
    A, rho, C = buckingham_params
    return A * np.exp(-r / rho) - C / r**6

# Numba-compiled function to compute the Buckingham matrix
@njit(parallel=True)
def compute_buckingham_matrix_numba(cart_coords, lattice_vectors, buckingham_matrix, buckingham_params_array, R_max, nx, ny, nz, N):
    for i in prange(N):
        for j in prange(i + 1, N):
            dr_init = cart_coords[i] - cart_coords[j]

            for rnx in prange(-nx, nx + 1):
                for rny in prange(-ny, ny + 1):
                    for rnz in prange(-nz, nz + 1):
                        shift = rnx * lattice_vectors[0] + rny * lattice_vectors[1] + rnz * lattice_vectors[2]
                        dr = dr_init + shift

                        if np.linalg.norm(dr) < R_max:
                            r = np.linalg.norm(dr)
                            buckingham_matrix[i][j] += buckingham_potential(buckingham_params_array[i, j], r)

# Main function that handles data preparation and non-Numba parts
def compute_buckingham_matrix(structure, buckingham_dict, R_max, max_shift=None):
    frac_coords = structure.frac_coords
    lattice_vectors = structure.lattice.matrix
    sites = structure.sites

    N = structure.num_sites
    cart_coords = frac_coords @ lattice_vectors
    
    if max_shift is None:
        max_real = max_lattice_translation(lattice_vectors, R_max)
        nx = max_real[0]
        ny = max_real[1]
        nz = max_real[2]
    else:
        ny = nz = nx = max_shift

    buckingham_matrix = np.zeros((N, N))
    
    # Preprocess buckingham parameters into a 2D array for Numba
    buckingham_params_array = np.zeros((N, N, 3))  # assuming 3 parameters: A, rho, and C
    
    for i in range(N):
        for j in range(i + 1, N):
            sites_label = f'{sites[i].specie}-{sites[j].specie}'
            if sites_label in buckingham_dict:
                buckingham_params_array[i, j] = buckingham_dict[sites_label]

    # Call the Numba-compiled function to compute the matrix
    compute_buckingham_matrix_numba(cart_coords, lattice_vectors, buckingham_matrix, buckingham_params_array, R_max, nx, ny, nz, N)
    
    return buckingham_matrix

#### THINK ABOUT FUNCTIONS BELOW

import numpy as np
import copy
import re

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from CRYSTALpytools.convert import cry_gui2pmg
from scipy import constants
k_b = constants.physical_constants['Boltzmann constant in eV/K'][0]

from dwave.embedding.chain_strength import  uniform_torque_compensation
from dwave.system import EmbeddingComposite, DWaveSampler
import minorminer
import dimod 

def add_num_dopant(dataframe,num_sites,dopant_species):
    config = dataframe.iloc[:,0:num_sites]
    n_dopant = np.sum(config==dopant_species,axis=1)
    dataframe['num_dopants'] = n_dopant
    
    return dataframe.sort_values(by='num_dopants')

def binomial_coefficient(n, k):
    return np.factorial(n) // (np.factorial(k) * np.factorial(n - k))

def adjacency_matrix_no_pbc(structure_pbc, max_neigh = 1, diagonal_terms = False, triu = False):
    # structure = pymatgen Structure object
    
    from pymatgen.core.structure import Molecule
    structure = Molecule(structure_pbc.atomic_numbers,structure_pbc.cart_coords)
    num_sites = structure.num_sites
    distance_matrix_pbc = np.round(structure.distance_matrix,5)

    distance_matrix = np.zeros((num_sites,num_sites),float)
    
    shells = np.unique(distance_matrix_pbc[0])
    
    for i,s in enumerate(shells[0:max_neigh+1]):
        row_index = np.where(distance_matrix_pbc == s)[0]
        col_index = np.where(distance_matrix_pbc == s)[1]
        distance_matrix[row_index,col_index] = i
    
    if triu == True:
        distance_matrix = np.triu(distance_matrix,0)
    
    if diagonal_terms == True:
        np.fill_diagonal(distance_matrix,[1]*num_sites)
    
    return distance_matrix

def build_binary_vector(atomic_numbers,atom_types=None):
    """Summary line.

    Extended description of function.

    Args:
        atomic_numbers (list): List of atom number of the sites in the structure
        atom_types (list): List of 2 elements. List element 0 = atomic_number of site == 0, 
                           list element 1 = atomic_number of site == 1

    Returns:
        List: Binary list of atomic numbers

    """
    
    atomic_numbers = np.array(atomic_numbers)
    num_sites = len(atomic_numbers)
    
    if atom_types == None:
        species = np.unique(atomic_numbers)
    else:
        species = atom_types
    
    binary_atomic_numbers = np.zeros(num_sites,dtype=int)
    
    for i,species_type in enumerate(species):
        #print(i,species_type)
        sites = np.where(atomic_numbers == species_type)[0]
        #print(i,species_type,sites)
        binary_atomic_numbers[sites] = i
    
    return binary_atomic_numbers

def build_ml_qubo(structure,X_train,y_train,max_neigh=1):
    
    #Filter
    distance_matrix = np.round(structure.distance_matrix,5)
    shells = np.unique(np.round(distance_matrix,5))
    num_sites = structure.num_sites
    distance_matrix_filter = np.zeros((num_sites,num_sites),int)

    for i,s in enumerate(shells[0:max_neigh+1]):
        row_index = np.where(distance_matrix == s)[0]
        col_index = np.where(distance_matrix == s)[1]
        distance_matrix_filter[row_index,col_index] = i
    distance_matrix_filter = np.triu(distance_matrix_filter,0)
    np.fill_diagonal(distance_matrix_filter,[1]*num_sites)
    
    #Build the descriptor

    upper_tri_indices = np.where(distance_matrix_filter != 0)
    descriptor = []

    for config in X_train:
        matrix = np.outer(config,config)
        upper_tri_elements = matrix[upper_tri_indices]
        descriptor.append(upper_tri_elements)
        

#     descriptor_all = []
#     for config in all_configurations:
#         matrix = np.outer(config,config)
#         upper_tri_elements = matrix[upper_tri_indices]
#         descriptor_all.append(upper_tri_elements)
    
    descriptor = np.array(descriptor)
    
    from sklearn.linear_model import LinearRegression
    
    
    reg = LinearRegression() #create the object
    reg.fit(descriptor, y_train)
    
    print('R2: ',reg.score(descriptor, y_train))

    Q = np.zeros((num_sites,num_sites))
    Q[upper_tri_indices] = reg.coef_
    
    return Q

def classical_energy(x,q):
    # x is the binary vector
    # q is the qubo matrix

    E_tmp = np.matmul(x,q)
    E_classical = np.sum(x*E_tmp)
    
    return E_classical

def extract_elastic_moduli(lines):
    # Find the start of the elastic moduli table
    start_idx = -1
    for i, line in enumerate(lines):
        if re.search(r'TOTAL ELASTIC MODULI \(kBar\)', line):
            start_idx = i
            break
    if start_idx == -1:
        raise ValueError("Elastic moduli section not found in the lines.")
    
    # Skip the first two lines (header and separator)
    data_lines = lines[start_idx+3:]
    
    matrix = []
    for line in data_lines:
        if re.match(r'\s*[-]+', line):  # Stop at the ending separator
            break
        # Split each line by spaces and filter out the first element (the direction label)
        values = list(map(float, line.split()[1:]))
        matrix.append(values)
    
    # Convert to a NumPy array
    return np.array(matrix)

#MAKE THIS GENERAL
def get_classical_av_conc(Q_gaaln_ml,mu_range,T,size=10000):


    size = size
    binary_vector = []
    QUBO_classical_E = []
    concentration = []

    # Generate binary vectors and compute classical energy
    for conc in np.random.randint(0, 54, size=size):
        concentration.append(conc)
        ones = np.random.choice(54, conc, replace=False)
        x = np.zeros(54, dtype='int')
        x[ones] = 1
        binary_vector.append(x)
        QUBO_classical_E.append(classical_energy(x, Q_gaaln_ml))

    # Convert lists to numpy arrays
    binary_vector = np.array(binary_vector)
    QUBO_classical_E = np.array(QUBO_classical_E)
    concentration = np.array(concentration)

    av_conc_classical = []
    mu_range = np.array(mu_range)
    mu_all = mu_range*Q_gaaln_ml[0][0]
    for mu in mu_all:
        #print(mu)
        for i in range(len(QUBO_classical_E)):
            #print(mu)
            energy_new = QUBO_classical_E + concentration*mu
        Z,pi = get_partition_function(energy_new,[1]*len(QUBO_classical_E),return_pi=True,T=T)
        av_conc_classical.append(np.sum(pi*concentration))
    av_conc_classical = np.array(av_conc_classical)/54
    
    return av_conc_classical

def get_nodes(problem,embedding):

    qpu_sampler = DWaveSampler(solver=dict(topology__type='pegasus'))
    qpu_graph = qpu_sampler.to_networkx_graph() 

    embedding = minorminer.find_embedding(problem,qpu_graph)

    data_dict = embedding

    # Create an empty set to store unique values
    unique_values_set = set()

    # Iterate through the values in the dictionary and add them to the set
    for values_list in data_dict.values():
        unique_values_set.update(values_list)

    # Convert the set back to a list to get unique values
    unique_values_list = list(unique_values_set)

    # Sort the list if needed
    unique_values_list.sort()
    
    max_length = 0
    #print(embedding)
    # Iterate through the values in the dictionary and update max_length if needed
    for values_list in data_dict.values():
        current_length = len(values_list)
        if current_length > max_length:
            max_length = current_length

    return len(unique_values_list),max_length

def get_partition_function(energy, multiplicity, T=298.15, return_pi=True, N_N=0, N_potential=0.):

    """
    Calculate the partition function and probabilities for different energy levels.
    
    Args:
        energy (np.ndarray): Array of energy levels.
        multiplicity (np.ndarray): Array of corresponding multiplicities.
        T (float, optional): Temperature in Kelvin. Default is 298.15 K.
        return_pi (bool, optional): Flag to return probabilities. Default is True.
        N_N (float, optional): Number of N particles. Default is 0.
        N_potential (float, optional): Potential for N particles. Default is 0.

    Returns:
        tuple or float: If return_pi is True, returns a tuple containing partition function and probabilities.
                        Otherwise, returns the partition function.
    """
    
    energy = np.array(energy)
    multiplicity = np.array(multiplicity)
    p_i = multiplicity * np.exp((-energy + (N_N * N_potential)) / (k_b * T))
    pf = np.sum(p_i)
    
    p_i /= pf
    
    if return_pi:
        return pf, p_i
    else:
        return pf
    
def get_qubo_energies(Q,all_configurations):
    
    predicted_energy = []
    
    for i,config in enumerate(all_configurations):
        predicted_energy.append(classical_energy(config,Q))
    
    return predicted_energy

def get_temperature(unique_energies,probability_energy,Z,mult, return_all = False):
    
    arr = -unique_energies/(k_b*(np.log((probability_energy*Z)/mult)))
    
    if return_all == True:
        return arr
    else:
        
        mask = ~np.isnan(arr)

        # Filter out NaN values using the mask
        arr_without_nan = arr[np.where(arr>0.)[0]]

        return np.average(arr_without_nan)

def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

def test_qubo_energies(y_pred,y_dft):
    
    from sklearn.metrics import mean_squared_error as mse
    
    return mse(y_pred, y_dft)


def test_qubo_energies_mape(y_pred,y_dft):
    
    from sklearn.metrics import mean_absolute_percentage_error as mse
    
    return mse(y_pred, y_dft)