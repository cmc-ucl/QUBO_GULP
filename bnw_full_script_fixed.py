import numpy as np
# import pandas as pd

import subprocess
import multiprocessing
import multiprocessing as mp


from ortools.sat.python import cp_model
from tqdm import tqdm
from pymatgen.core.structure import Structure, PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.periodic_table import Element
from pymatgen.io.cif import *
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core.lattice import Lattice

try:
    from full_script_functions import write_gulp_input as _default_write_gulp_input
except ImportError:  # fallback when helper module is unavailable
    _default_write_gulp_input = None

from scipy.spatial.distance import pdist, squareform

from ase.visualize import view
from ase.io import write, read


from pymatgen.io.ase import AseAtomsAdaptor

import os, json, gzip, hashlib, time, textwrap


import copy


#import dataframe_image as dfi

from scipy import constants
from scipy.spatial import cKDTree

# import matplotlib.pyplot as plt


k_b = constants.physical_constants['Boltzmann constant in eV/K'][0]
# print(k_b)
def vview(structure):
    view(AseAtomsAdaptor().get_atoms(structure))

np.seterr(divide='ignore')
# plt.style.use('tableau-colorblind10')

# import seaborn as sns
import time


class StreamingIncumbentSaver(cp_model.CpSolverSolutionCallback):
    def __init__(self, x, site_options, li_sites, mn_sites, scale, out_dir, limit=None):
        super().__init__()
        self.x = x
        self.site_options = site_options
        self.li_sites = li_sites
        self.mn_sites = mn_sites
        self.scale = scale
        self.out_dir = out_dir
        self.limit = limit
        self.count = 0
        os.makedirs(out_dir, exist_ok=True)
        self.inc_path = os.path.join(out_dir, "incumbents.jsonl.gz")

    def on_solution_callback(self):
        if self.limit is not None and self.count >= self.limit:
            return
        # Decode current incumbent
        assignment = {s: next(a for a in opts if self.Value(self.x[(s,a)]) == 1)
                      for s, opts in self.site_options.items()}
        E = None if self.ObjectiveValue() is None else self.ObjectiveValue() / self.scale

        # Minimal record (same shape as append_incumbent)
        li_on  = sorted(int(s) for s in self.li_sites if assignment[s] == "Li")
        mn3_on = sorted(int(s) for s in self.mn_sites if assignment[s] == "Mn3")
        cfg_bytes = json.dumps({"li_on": li_on, "mn3_on": mn3_on}, separators=(",", ":")).encode()
        cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()[:16]

        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "E": E, "li_on": li_on, "mn3_on": mn3_on,
            "n_li": len(li_on), "n_mn3": len(mn3_on),
            "cfg": cfg_hash, "tags": {"status": "INCUMBENT"}
        }
        with gzip.open(self.inc_path, "ab") as gz:
            gz.write((json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8"))

        self.count += 1


def add_li_mn_charge_balance_constraints(model: cp_model.CpModel, x, li_sites, mn_sites, N_li: int):
    """
    Enforces:
      - total number of Li atoms == N_li
      - total number of Mn3+ ions == N_li
    """
    # Li count constraint
    model.Add(sum(x[(s, "Li")] for s in li_sites) == N_li)

    # Mn3+ count constraint
    model.Add(sum(x[(s, "Mn3")] for s in mn_sites) == N_li)


def add_li_proximity_exclusions(model: cp_model.CpModel, x, proximity_groups):
    """
    Enforce Li–Li exclusion constraints.

    Parameters
    ----------
    model : cp_model.CpModel
        The model to which constraints are added.
    x : dict
        Variable dictionary {(site_id, option): BoolVar}.
    proximity_groups : list
        Each element is either:
            - a tuple/list of two site IDs (s, t)  → pairwise exclusion
            - a list/tuple of ≥2 site IDs forming a clique (all mutually too close)

    Effect
    ------
    For every group g of sites, ensures that at most one can be occupied by Li:
        sum_{s∈g} x[s, "Li"] ≤ 1
    Uses CP-SAT’s AddAtMostOne for stronger propagation.
    """
    num_groups = 0
    for g in proximity_groups:
        # Normalise input
        if isinstance(g, tuple) or isinstance(g, list):
            sites = list(g)
        else:
            raise ValueError("Each proximity group must be a list/tuple of site IDs.")
        if len(sites) < 2:
            continue

        # Collect the BoolVars for these sites
        li_vars = [x[(s, "Li")] for s in sites]
        model.AddAtMostOne(li_vars)
        num_groups += 1


def add_ut_qubo_objective(
    model: cp_model.CpModel,
    x: dict,                  # {(site_id, option_name): BoolVar}
    var2siteopt: dict,        # {qubo_col_index: (site_id, option_name)}
    Q_ut: np.ndarray,         # upper-triangular QUBO matrix (shape n x n)
    *,
    scale: float = 1000.0,    # integer scaling for CP-SAT
    tiny: float = 1e-12,
    name_prefix: str = "y"
):
    """
    Add a minimization objective equivalent to an *upper-triangular* QUBO.

    Energy = sum_i     Q[i,i] * x_(s,a)
           + sum_{i<j} Q[i,j] * (x_(s,a) AND x_(t,b))

    where Q's columns/rows index your original binary vars (Li, Mn4, Mn3),
    and var2siteopt maps each original var index -> (site, option).

    Notes:
    - We only iterate i<=j because Q is upper-triangular (no double counting).
    - Same-site off-diagonals (i<j but s==t) are skipped (redundant under one-hot).
    - Coefficients are scaled to integers for CP-SAT.
    """
    n = Q_ut.shape[0]

    # 1) Integerize (keep upper triangle semantics)
    Q = np.array(Q_ut, dtype=float, copy=True)
    Qi = np.rint(Q * scale).astype(int)
    # prune tiny
    Qi[np.abs(Qi) < tiny] = 0
    SCALE = int(scale)

    obj_terms = []
    num_diag_added = 0
    num_pairs_added = 0
    num_pairs_skipped_same_site = 0
    num_pairs_skipped_zero = 0
    num_pairs_total_seen = 0

    # 2) Diagonal: Q[i,i] * x_(s,a)
    for i in range(n):
        if i not in var2siteopt:
            continue
        s, a = var2siteopt[i]
        c = Qi[i, i]
        if c != 0:
            obj_terms.append(c * x[(s, a)])
            num_diag_added += 1

    # 3) Off-diagonals (upper triangle): Q[i,j] * y, with y = AND(x_(s,a), x_(t,b))
    for i in range(n):
        if i not in var2siteopt:
            continue
        s, a = var2siteopt[i]
        row = Qi[i]
        for j in range(i + 1, n):      # i<j, upper-triangular entries only
            num_pairs_total_seen += 1
            c = row[j]
            if c == 0:
                num_pairs_skipped_zero += 1
                continue
            if j not in var2siteopt:
                continue
            t, b = var2siteopt[j]
            if s == t:
                # cross-terms within the same physical site are redundant with one-hot
                num_pairs_skipped_same_site += 1
                continue

            y = model.NewBoolVar(f"{name_prefix}_{s}_{a}_{t}_{b}")
            model.Add(y <= x[(s, a)])
            model.Add(y <= x[(t, b)])
            model.Add(y >= x[(s, a)] + x[(t, b)] - 1)
            obj_terms.append(c * y)
            num_pairs_added += 1

    # 4) Set objective
    model.Minimize(sum(obj_terms))

    # 5) Diagnostics
    summary = {
        "num_diag_added": num_diag_added,
        "num_pairs_total_seen": num_pairs_total_seen,
        "num_pairs_added": num_pairs_added,
        "num_pairs_skipped_zero": num_pairs_skipped_zero,
        "num_pairs_skipped_same_site": num_pairs_skipped_same_site,
        "scale": SCALE,
    }
    return SCALE, summary


def append_incumbent(
    output_dir: str,
    assignment: dict,            # {site_id: option_name}
    energy_ev: float | None,
    *,
    li_sites: list,
    mn_sites: list,
    tags: dict = None           # optional metadata, e.g. {"status":"FINAL"}
):
    """Append one incumbent configuration to incumbents.jsonl.gz."""
    li_on = sorted(int(s) for s in li_sites if assignment[s] == "Li")
    mn3_on = sorted(int(s) for s in mn_sites if assignment[s] == "Mn3")

    # Stable short hash for deduplication
    cfg_bytes = json.dumps({"li_on": li_on, "mn3_on": mn3_on},
                           separators=(",", ":")).encode("utf-8")
    cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()[:16]

    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "E": None if energy_ev is None else float(energy_ev),
        "li_on": li_on,
        "mn3_on": mn3_on,
        "n_li": len(li_on),
        "n_mn3": len(mn3_on),
        "cfg": cfg_hash,
    }
    if tags:
        rec["tags"] = tags

    inc_path = os.path.join(output_dir, "incumbents.jsonl.gz")
    with gzip.open(inc_path, "ab") as gz:
        gz.write((json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8"))

    return cfg_hash


def append_unique_extxyz(extxyz_path, structures, energies, tol_frac=1e-4, tol_lat=1e-3):
    """
    Append unique pymatgen.Structure entries to an extxyz file.
    - Creates the file if it doesn't exist.
    - Deduplicates vs existing frames AND within this batch.
    - Stores per-structure energy in Atoms.info['energy'].

    Returns: (n_added, n_skipped_existing, n_skipped_within_batch)
    """
    adaptor = AseAtomsAdaptor()

    # 1) Build existing hash set (if file exists)
    existing_hashes = set()
    if os.path.exists(extxyz_path) and os.path.getsize(extxyz_path) > 0:
        try:
            for atoms in read(extxyz_path, index=":"):
                # Rebuild a pmg Structure to reuse same hashing
                pmg = adaptor.get_structure(atoms)
                existing_hashes.add(structure_hash_pbc(pmg, tol_frac, tol_lat))
        except Exception:
            # If the extxyz is huge or partially corrupted, you can decide to ignore
            pass

    # 2) Filter incoming by hash (against existing and within-batch)
    batch_hashes = set()
    unique_atoms = []
    skipped_existing = 0
    skipped_batch = 0

    for s, e in zip(structures, energies):
        h = structure_hash_pbc(s, tol_frac, tol_lat)
        if h in existing_hashes:
            skipped_existing += 1
            continue
        if h in batch_hashes:
            skipped_batch += 1
            continue

        a = adaptor.get_atoms(s)
        a.info["energy"] = float(e) if e is not None else None
        unique_atoms.append(a)
        batch_hashes.add(h)

    # 3) Append to file (or create)
    if unique_atoms:
        write(extxyz_path, unique_atoms, append=os.path.exists(extxyz_path))

    return len(unique_atoms), skipped_existing, skipped_batch


def average_close_points(symmetrised_coords, threshold):
    """
    Averages the coordinates of points that are closer than the given threshold.

    Parameters:
    symmetrised_coords (np.array): (N,3) array of 3D coordinates.
    threshold (float): Distance threshold for grouping points.

    Returns:
    np.array: New array with averaged coordinates.
    """
    # Compute pairwise distances
    distances = squareform(pdist(symmetrised_coords))
    
    # Track processed points
    visited = np.zeros(len(symmetrised_coords), dtype=bool)
    averaged_coords = []

    for i in range(len(symmetrised_coords)):
        if visited[i]:  # Skip if already processed
            continue

        # Find all close points (including itself)
        close_points = np.where(distances[i] < threshold)[0]
        visited[close_points] = True  # Mark as processed

        # Compute the average of these points
        avg_coord = np.mean(symmetrised_coords[close_points], axis=0)
        averaged_coords.append(avg_coord)

    return np.array(averaged_coords)


def build_li_proximity_groups(
    li_grid_coords,
    threshold_ang,
    *,
    lattice=None,
    coords_are_cartesian=True,
    site_ids=None,
    return_pairs_also=True,
):
    """
    Build Li–Li proximity groups for CP-SAT 'AtMostOne' constraints.

    Parameters
    ----------
    li_grid_coords : (M,3) array
        Li candidate site coordinates. If 'coords_are_cartesian' is False,
        they are treated as fractional.
    threshold_ang : float
        Distance threshold (Å) for "too close".
    lattice : (3,3) array-like or pymatgen Lattice, optional
        Required if periodic MIC distances matter. If None and
        coords_are_cartesian=True, uses plain Euclidean distances (no PBC).
    coords_are_cartesian : bool
        Whether li_grid_coords are in Cartesian. If True and lattice is given,
        MIC distances are used. If False, coords are treated as fractional.
    site_ids : list[int], optional
        CP site IDs aligned with li_grid_coords. If None, uses range(M).
    return_pairs_also : bool
        If True, also return the raw list of close pairs.

    Returns
    -------
    groups : list[list[int]]
        Each group is a clique (size ≥ 2) of site IDs that are mutually
        within the threshold (use one AddAtMostOne per group).
    pairs  : list[tuple[int,int]]  (only if return_pairs_also=True)
        All offending pairs (site_i, site_j) by ID.
    """
    coords = np.asarray(li_grid_coords, dtype=float)
    M = coords.shape[0]
    if site_ids is None:
        site_ids = list(range(M))

    # Build fractional coords (needed for MIC)
    if coords_are_cartesian:
        if lattice is None:
            # no PBC: simple Euclidean threshold
            # build edges directly
            edges = []
            for i in range(M):
                for j in range(i+1, M):
                    if np.linalg.norm(coords[j] - coords[i]) < threshold_ang:
                        edges.append((i, j))
        else:
            L = lattice.matrix if hasattr(lattice, "matrix") else np.asarray(lattice, dtype=float)
            f = _frac_coords(coords, L)
            edges = _pair_edges_with_threshold(f % 1.0, L, threshold_ang)
    else:
        # coords already fractional
        if lattice is None:
            raise ValueError("lattice is required when using fractional coords to compute distances.")
        L = lattice.matrix if hasattr(lattice, "matrix") else np.asarray(lattice, dtype=float)
        edges = _pair_edges_with_threshold(coords % 1.0, L, threshold_ang)

    # Maximal cliques from the proximity graph
    cliques = _maximal_cliques_from_edges(M, edges)

    # Map internal indices -> site_ids
    groups = [[site_ids[i] for i in clique] for clique in cliques]
    if return_pairs_also:
        pairs = [(site_ids[i], site_ids[j]) for (i, j) in edges]
        return groups, pairs
    return groups


def build_new_structural_model(opt_structures, M, N_positions_final, initial_structure, threshold, return_stats = False, fix_angles=True):


    if fix_angles == True:
        # --- Average only the lattice lengths, keep angles fixed from the reference structure ---
        alpha = 90
        beta = 90
        gamma = 90

        a_vals, b_vals, c_vals = [], [], []
        for s in opt_structures:
            a, b, c = s.lattice.lengths
            a_vals.append(a)
            b_vals.append(b)
            c_vals.append(c)

        a_mean = np.mean(a_vals)
        b_mean = np.mean(b_vals)
        c_mean = np.mean(c_vals)

        lattice_new = Lattice.from_parameters(
            a_mean,
            b_mean,
            c_mean,
            alpha,
            beta,
            gamma
        )

    else:
        lattice_all = []
        for structure in opt_structures:
            lattice_all.append(structure.lattice.matrix)
        lattice_new = np.mean(lattice_all,axis=0)

    mn_coord_new = []
    o_coord_new = []

    for structure in opt_structures:
        work = structure.copy()
        work.replace_species({'Tc':'Mn'})
        mn_indices_new = np.where(np.array(work.atomic_numbers)==25)[0]
        o_indices_new = np.where(np.array(work.atomic_numbers)==8)[0]

        mn_coord_new.append(work.frac_coords[mn_indices_new]%1)
        o_coord_new.append(work.frac_coords[o_indices_new]%1)

    mn_coord_new = unwrap_frac_coords(mn_coord_new)
    o_coord_new = unwrap_frac_coords(o_coord_new)

    mn_coord_new = np.array(mn_coord_new)
    o_coord_new = np.array(o_coord_new)

    mn_coord_average = np.mean(mn_coord_new,axis=0)
    o_coord_average = np.mean(o_coord_new,axis=0)
    if return_stats == True:

        mn_coord_average_std = np.average(np.std(mn_coord_new,axis=0))
        o_coord_average_std = np.average(np.std(o_coord_new,axis=0))

        mn_coord_max_std = np.average(np.max(mn_coord_new,axis=0))
        o_coord_max_std = np.average(np.max(o_coord_new,axis=0))

    
    #Lithium

    li_coords_all = []

    for structure in opt_structures:
        work = structure.copy()
        work.replace_species({'Tc':'Mn'})
        
        li_index = np.where(np.array(work.atomic_numbers) == 3)[0]

        li_coords = work.frac_coords[li_index]
        li_coords_all.extend(li_coords)
            
    li_coords_all = unwrap_frac_coords(li_coords_all)

    li_coords_all = np.array(li_coords_all)
    grid = compute_probability_grid(li_coords_all, M)

    # plot_probability_grid(grid)

    centers = find_fractional_centers(M)
    top_centers = find_top_x_points(grid,centers,N_positions_final)

    coord_top = []
    for line in top_centers:
        coord_top.append(line[0])
    coord_top = np.array(coord_top)

    #Symmetrise
    symmops = SpacegroupAnalyzer(initial_structure).get_symmetry_operations()
    num_symmops = len(symmops)

    symmetrised_coords = []
    for symmop in symmops:
        for coord in coord_top:
            symmetrised_coords.append(symmop.operate(coord)%1)


    symmetrised_coords = np.array(symmetrised_coords)

    averaged_symmetrised_coords = average_close_points(symmetrised_coords, threshold)
    # vview(Structure(initial_structure.lattice.matrix,[1]*N_positions_final,coord_top))
    # vview(Structure(initial_structure.lattice.matrix,[1]*len(averaged_symmetrised_coords),averaged_symmetrised_coords))

    mn_sites = []
    o_sites = []
    li_sites = []

    # atomic_numbers = [3]*len(averaged_symmetrised_coords)+[24]*len(mn_coord_new)+[8]*

    li_sites = []
    mn_sites = []
    o_sites = []

    lattice = lattice_new if isinstance(lattice_new, Lattice) else Lattice(lattice_new)

    # Then use it everywhere:
    li_sites = [PeriodicSite('Li', coord, lattice) for coord in averaged_symmetrised_coords]
    mn_sites = [PeriodicSite('Mn', coord, lattice) for coord in mn_coord_average]
    o_sites = [PeriodicSite('O', coord, lattice) for coord in o_coord_average]

    # Combine and build
    all_sites = li_sites + mn_sites + o_sites
    structure = Structure.from_sites(all_sites)

    return structure, averaged_symmetrised_coords


def build_QUBO(structure, threshold_li=0, prox_penalty=0):
    
    structure_tmp = copy.deepcopy(structure)
    structure_tmp.add_site_property("charge", [1.0] * len(structure))
    # ewald_matrix = compute_ewald_matrix_fast(structure,triu=True)
    ewald = EwaldSummation(structure_tmp, eta=None, w=1)

    ewald_matrix = ewald.total_energy_matrix
    ewald_matrix = np.triu(ewald_matrix,1)


    charges = {
        25: [4,3],
        3: [1],
        8: [-2]
    }

    if threshold_li > 0 and prox_penalty > 0:
        # THE PROX PENALTY WILL BE MULTIPLIED BY THE CHARGES (1 IN THIS CASE)
        # Add contstraint on proximity of lithium atoms
        dm = structure.distance_matrix
        num_sites = structure.num_sites
        li_indices = np.where(np.array(structure.atomic_numbers) == 3)[0]
        num_o = np.sum(np.array(structure.atomic_numbers) == 8)
        
        # Create a mask for all (i,j) pairs where both i and j are Li
        li_mask = np.zeros((num_sites, num_sites), dtype=bool)
        li_mask[np.ix_(li_indices, li_indices)] = True

        # Apply the distance threshold
        below_thresh_mask = dm < threshold_li

        # Combine masks
        final_mask = li_mask & below_thresh_mask

        # Create constraint matrix
        prox_constraint = np.where(final_mask, prox_penalty, 0)
        np.fill_diagonal(prox_constraint,0)

        ewald_matrix += prox_constraint
  
    ewald_discrete, expanded_charges, expanded_matrix = compute_discrete_ewald_matrix(structure, charges, ewald_matrix)

    species_dict = {'Mn': ['Mn', 'Tc']}  # Mn sites can be either Mn4+ (Mn) or Mn3+ (Tc)
    buckingham_dict = {'Li-O':[426.480 ,    0.3000  ,   0.00],
                        'Mn-O':[3087.826    ,   0.2642 ,    0.00], # This is the Mn4+
                        'Tc-O':[1686.125  ,    0.2962 ,    0.00], # This is the Mn3+
                        'O-O' : [22.410  ,     0.6937,   32.32]
                        }
    buckingham_discrete, species_vector = compute_buckingham_matrix_discrete_parallel(
        structure, species_dict, buckingham_dict, R_max=25.0
    )

    Q_discrete = build_qubo_discrete_from_Ewald_IP(ewald_discrete,buckingham_discrete)
    
    # === Create mask to remove 'O' sites ===
    mask = [el != 'O' for el in species_vector]

    # Apply mask to species vector
    reduced_species_vector = [el for el, keep in zip(species_vector, mask) if keep]

    # Apply mask to QUBO matrix
    QUBO, oo_energy = reduce_qubo_discrete_limno(Q_discrete, species_vector)

    # Compute correct indices based on reduced species vector
    li_indices = [i for i, el in enumerate(reduced_species_vector) if el == 'Li']
    mn_indices = [i for i, el in enumerate(reduced_species_vector) if el in ('Mn', 'Tc')]


    # # THIS IS A QUICK FIX THAT ONLY WORKS IF THE ATOMS ARE IN ORDER Mn-O-Li
    # li_indices = np.array([i for i, el in enumerate(species_vector) if el == 'Li']) - num_o
    # li_indices = li_indices.tolist()
 
    # mn_indices = [i for i, el in enumerate(species_vector) if el in ('Mn', 'Tc')]

    return QUBO, li_indices, mn_indices


def build_qubo_discrete_from_Ewald_IP(ewald_discrete,buckingham_matrix):
    Q = ewald_discrete + buckingham_matrix

    return Q


def build_site_option_maps_from_indices(li_indices, mn_indices):
    """
    Input:
      li_indices: list[int]  -> QUBO columns that mean 'Li present'
      mn_indices: list[int]  -> [Mn4_0, Mn3_0, Mn4_1, Mn3_1, ...]
    Output:
      site_options: dict[site_id] -> list[str] of options
      var2siteopt: dict[qubo_col] -> (site_id, option_name)
      li_sites: list[int] of site_ids that are Li grid sites
      mn_sites: list[int] of site_ids that are Mn sites
    """
    assert len(mn_indices) % 2 == 0, "mn_indices must be even length (pairs)."

    site_options = {}
    var2siteopt  = {}
    li_sites, mn_sites = [], []
    print(len(li_indices))
    # 1) Li grid sites: create a site with options ["Empty","Li"] for each li_index
    for k in li_indices:
        s = len(site_options)
        site_options[s] = ["Empty", "Li"]
        var2siteopt[k]  = (s, "Li")      # the QUBO var corresponds to the "Li" option
        li_sites.append(s)
    print(len(mn_indices))
    # 2) Mn sites: every consecutive pair -> one Mn site with ["Mn4","Mn3"]
    for p in range(0, len(mn_indices), 2):
        k4 = mn_indices[p]
        k3 = mn_indices[p+1]
        s = len(site_options)
        site_options[s] = ["Mn4", "Mn3"]
        var2siteopt[k4] = (s, "Mn4")
        var2siteopt[k3] = (s, "Mn3")
        mn_sites.append(s)

    return site_options, var2siteopt, li_sites, mn_sites


def build_x_vars_and_onehot(model: cp_model.CpModel, site_options):
    """
    Make BoolVars x[(s,a)] and add one-hot per site: sum_a x[s,a] == 1
    Returns: x dict
    """
    x = {}
    for s, opts in site_options.items():
        for a in opts:
            x[(s, a)] = model.NewBoolVar(f"x_{s}_{a}")
        # one-hot: exactly one option per site
        model.Add(sum(x[(s, a)] for a in opts) == 1)
    return x


def compute_buckingham_matrix_discrete(structure, species_dict, buckingham_dict, R_max, max_shift=None,
                                       distance_analysis=False, distance_threshold=0.1):
    """
    Compute an expanded Buckingham potential matrix for a system where certain chemical species
    can exist as multiple elements (e.g., 'Ca' → ['Mg', 'Ca']).

    Parameters
    ----------
    structure : pymatgen.Structure
        The atomic structure.
    species_dict : dict
        Dictionary mapping species labels (str) to possible alternative elements.
        Example: {'Ca': ['Mg', 'Ca']}.
    buckingham_dict : dict
        Dictionary of Buckingham parameters for each element pair (e.g., "Ca-F").
    R_max : float
        Maximum real space cutoff.
    max_shift : int, optional
        Maximum lattice vector translation in each direction.
    distance_analysis : bool
        If True, will flag very short distances.
    distance_threshold : float
        Threshold for flagging short distances.

    Returns
    -------
    buckingham_matrix_expanded : np.ndarray
        The expanded Buckingham interaction matrix.
    expanded_species : list of str
        Species labels corresponding to the rows/columns of the matrix.
    """
    import numpy as np
    from pymatgen.core.periodic_table import Element
    from tqdm import tqdm

    def buckingham_potential(params, r):
        A, rho, C = params
        return A * np.exp(-r / rho) - C / r**6 if r != 0 else 0

    frac_coords = structure.frac_coords
    lattice_vectors = structure.lattice.matrix
    cart_coords = frac_coords @ lattice_vectors
    distance_matrix = structure.distance_matrix
    sites = structure.sites

    N = len(structure)
    index_map = {}
    expanded_species = []
    new_idx = 0

    # Step 1: Build index mapping and expanded species list
    for i, site in enumerate(sites):
        sp = str(site.specie)
        if sp in species_dict:
            options = species_dict[sp]
            index_map[i] = list(range(new_idx, new_idx + len(options)))
            expanded_species.extend(options)
            new_idx += len(options)
        else:
            index_map[i] = [new_idx]
            expanded_species.append(sp)
            new_idx += 1

    expanded_N = len(expanded_species)
    buckingham_matrix_expanded = np.zeros((expanded_N, expanded_N))

    # Step 2: Determine max lattice shift
    if max_shift is None:
        max_real = np.ceil(R_max / np.linalg.norm(lattice_vectors, axis=1)).astype(int)
        nx, ny, nz = max_real
    else:
        nx = ny = nz = max_shift

    # Step 3: Fill the expanded matrix
    for i in tqdm(range(N), desc="Buckingham matrix"):
        for j in range(i + 1, N):
            sp_i = str(sites[i].specie)
            sp_j = str(sites[j].specie)
            options_i = species_dict.get(sp_i, [sp_i])
            options_j = species_dict.get(sp_j, [sp_j])
            dr_init = cart_coords[i] - cart_coords[j]
            dr_dm = distance_matrix[i][j]

            for ii, ei in zip(index_map[i], options_i):
                for jj, ej in zip(index_map[j], options_j):
                    pair_key1 = f"{ei}-{ej}"
                    pair_key2 = f"{ej}-{ei}"
                    key = pair_key1 if pair_key1 in buckingham_dict else pair_key2 if pair_key2 in buckingham_dict else None
                    if not key:
                        continue

                    if distance_analysis and dr_dm < distance_threshold:
                        buckingham_matrix_expanded[ii, jj] = 1e6
                    else:
                        for rnx in range(-nx, nx + 1):
                            for rny in range(-ny, ny + 1):
                                for rnz in range(-nz, nz + 1):
                                    shift = rnx * lattice_vectors[0] + rny * lattice_vectors[1] + rnz * lattice_vectors[2]
                                    dr = dr_init + shift
                                    dist = np.linalg.norm(dr)
                                    if dist < R_max:
                                        V = buckingham_potential(buckingham_dict[key], dist)
                                        buckingham_matrix_expanded[ii, jj] += V

    return buckingham_matrix_expanded, expanded_species


def compute_buckingham_matrix_discrete_fast(
    structure,
    species_dict,
    buckingham_dict,
    R_max,
    distance_analysis=False,
    distance_threshold=0.1,
):
    """
    Faster Buckingham matrix builder:
      - Uses get_points_in_sphere (spherical neighbor enumeration, no box scan)
      - Vectorizes over all periodic images for each (i,j)
      - Reuses sums per unique rho and a global sum of 1/r^6
    """

    sites = structure.sites
    N = len(sites)
    lat = structure.lattice
    fcoords = structure.frac_coords
    ccoords = structure.cart_coords  # only for initial neighbor seeds

    # ---- 0) Preprocess species options and index map (same as your code) ----
    index_map = {}
    expanded_species = []
    new_idx = 0
    for i, site in enumerate(sites):
        sp = str(site.specie)
        options = species_dict.get(sp, [sp])
        index_map[i] = list(range(new_idx, new_idx + len(options)))
        expanded_species.extend(options)
        new_idx += len(options)
    expanded_N = len(expanded_species)
    B = np.zeros((expanded_N, expanded_N), dtype=float)

    # ---- 1) Preprocess Buckingham parameters & unique rhos ----
    # Normalize keys to canonical "A-B" with A<=B
    def canon_pair(a, b):
        return (a, b) if a <= b else (b, a)

    params = {}       # (ei,ej) -> (A, rho, C)
    unique_rhos = set()
    for key, (A, rho, C) in buckingham_dict.items():
        Ael, Bel = key.replace(" ", "").split("-")
        k = canon_pair(Ael, Bel)
        params[k] = (float(A), float(rho), float(C))
        unique_rhos.add(float(rho))
    unique_rhos = sorted(unique_rhos)

    # ---- 2) Main loop: i<j, gather all periodic images within R_max once ----
    # For each (i,j), compute
    #   S_r6   = sum_k (1/r_k^6)
    #   S_exp[ρ] = sum_k exp(-r_k / ρ)  for each unique ρ
    # and then fill all option pairs via parameters lookup.
    for i in tqdm(range(N), desc="Buckingham (fast)"):
        ri = ccoords[i]
        spi = str(sites[i].specie)
        opts_i = species_dict.get(spi, [spi])
        inds_i = index_map[i]

        # neighbor search around ri using *fractional* seed list fcoords
        # get_points_in_sphere expects fractional list + cartesian center
        nf, dists, js, _imgs = lat.get_points_in_sphere(fcoords, ri, R_max, zip_results=False)

        # Strip self-image at r≈0
        mask = dists > 1e-12
        nf = nf[mask]; dists = dists[mask]; js = js[mask]
        if len(dists) == 0:
            continue

        # Precompute aggregates once per i for all j (we’ll mask per j below)
        # These are *per-(i,j)* actually, so we compute per j mask shortly.

        # Group contributions by j to avoid scanning full arrays for each rho repeatedly
        # Build an index list for each distinct j in neighbors
        js_unique, inv = np.unique(js, return_inverse=True)
        # For each block corresponding to a single j, precompute S_r6 and S_exp[ρ]
        for idx, j in enumerate(js_unique):
            if j <= i:
                continue  # keep upper triangle & avoid double counting
            block_mask = (inv == idx)
            dj = dists[block_mask]
            if dj.size == 0:
                continue

            # distance-based screening
            if distance_analysis and dj.min() < distance_threshold:
                # Apply huge penalty to all option pairs of (i,j)
                for ii in inds_i:
                    for jj in index_map[j]:
                        B[ii, jj] += 1e6
                        B[jj, ii] += 1e6
                continue

            # vectorized aggregates for this (i,j)
            inv_r6 = (dj**-6).sum()               # S_r6
            Sexp_by_rho = {rho: np.exp(-dj / rho).sum() for rho in unique_rhos}

            spj = str(sites[j].specie)
            opts_j = species_dict.get(spj, [spj])
            inds_j = index_map[j]

            # Fill contributions for all option pairs
            for ii, ei in zip(inds_i, opts_i):
                for jj, ej in zip(inds_j, opts_j):
                    A, rho, C = params.get(canon_pair(ei, ej), (None, None, None))
                    if A is None:
                        continue
                    Vij = A * Sexp_by_rho[rho] - C * inv_r6
                    B[ii, jj] += Vij
                    B[jj, ii] += Vij   # symmetric

    return B, expanded_species


def compute_buckingham_matrix_discrete_parallel(
    structure,
    species_dict,
    buckingham_dict,
    R_max,
    distance_analysis=False,
    distance_threshold=0.1,
):
    """
    Parallel fast Buckingham matrix builder.
    """

    sites = structure.sites
    N = len(sites)
    lat = structure.lattice
    fcoords = structure.frac_coords
    ccoords = structure.cart_coords

    # --- Species expansion
    index_map = {}
    expanded_species = []
    idx = 0
    for i, site in enumerate(sites):
        sp = str(site.specie)
        opts = species_dict.get(sp, [sp])
        index_map[i] = list(range(idx, idx + len(opts)))
        expanded_species.extend(opts)
        idx += len(opts)
    expanded_N = len(expanded_species)

    # --- Preprocess Buckingham params
    def canon_pair(a, b):
        return (a, b) if a <= b else (b, a)

    buckingham_params = {}
    unique_rhos = set()
    for key, (A, rho, C) in buckingham_dict.items():
        a, b = key.replace(" ", "").split("-")
        pair = canon_pair(a, b)
        buckingham_params[pair] = (float(A), float(rho), float(C))
        unique_rhos.add(float(rho))
    unique_rhos = sorted(unique_rhos)

    # --- Build work chunks
    ncpu = mp.cpu_count()
    chunk = (N + ncpu - 1) // ncpu
    tasks = []
    for k in range(ncpu):
        i_start = k * chunk
        i_end = min((k + 1) * chunk, N)
        if i_start >= i_end:
            continue

        args = (
            i_start, i_end,
            fcoords, ccoords, sites, lat,
            species_dict, buckingham_params, unique_rhos, R_max,
            distance_analysis, distance_threshold,
            index_map, expanded_N
        )
        tasks.append(args)

    # --- Run in parallel
    print(f"Launching {len(tasks)} parallel workers over {ncpu} CPUs...")
    with mp.Pool(len(tasks)) as pool:
        partial_results = list(
            tqdm(pool.imap(_buckingham_worker, tasks), total=len(tasks))
        )

    # --- Sum partial matrices
    B = np.sum(partial_results, axis=0)

    return B, expanded_species



def compute_discrete_ewald_matrix(structure, charge_options_by_Z, ewald_matrix):
    """
    Computes an expanded charge-weighted Ewald matrix for a pymatgen.Structure
    using a dictionary of charge options by atomic number.

    Parameters
    ----------
    structure : pymatgen.Structure
        The atomic structure.
    charge_options_by_Z : dict
        Dictionary mapping atomic numbers (Z) to lists of possible charges.
        Example: {26: [2, 3]} for Fe²⁺ and Fe³⁺.
    ewald_matrix : np.ndarray, optional
        Precomputed Ewald matrix. If None, will call compute_ewald_matrix(structure).

    Returns
    -------
    weighted_ewald : np.ndarray
        The Ewald matrix weighted by the outer product of the expanded charges.
    expanded_charges : np.ndarray
        1D array of charges including duplicated sites.
    expanded_ewald_matrix : np.ndarray
        Expanded Ewald matrix matching the length of `expanded_charges`.
    """

    num_sites = len(structure)
    atomic_numbers = structure.atomic_numbers

    total_new_sites = sum(
        (len(charge_options_by_Z[Z]) - 1) * np.sum(np.array(atomic_numbers) == Z)
        for Z in charge_options_by_Z
        if Z in atomic_numbers
    )

    expanded_N = num_sites + total_new_sites
    expanded_matrix = np.zeros((expanded_N, expanded_N))
    expanded_charges = []
    index_map = {}

    new_idx = 0
    for i, Z in enumerate(atomic_numbers):
        if Z in charge_options_by_Z:
            possible_charges = charge_options_by_Z[Z]
            index_map[i] = list(range(new_idx, new_idx + len(possible_charges)))
            expanded_charges.extend(possible_charges)
            new_idx += len(possible_charges)
        else:
            # Default to site charge if it exists, else zero
            try:
                default_charge = structure[i].properties.get("charge", 0)
            except AttributeError:
                default_charge = 0
            index_map[i] = [new_idx]
            expanded_charges.append(default_charge)
            new_idx += 1

    expanded_charges = np.array(expanded_charges)

    # Expand the Ewald matrix based on duplication map
    for i in range(num_sites):
        for j in range(num_sites):
            for ii in index_map[i]:
                for jj in index_map[j]:
                    expanded_matrix[ii, jj] = ewald_matrix[i, j]

    charge_matrix = np.outer(expanded_charges, expanded_charges)
    weighted_ewald = expanded_matrix * charge_matrix

    return weighted_ewald, expanded_charges, expanded_matrix


def compute_probability_grid(li_coords_all, M):
    """
    Compute a MxMxM grid of probabilities for the points in li_coords_all.
    
    Parameters:
    li_coords_all (ndarray): Nx3 array of fractional coordinates.
    M (int): Size of the grid along each dimension.
    
    Returns:
    ndarray: MxMxM array of probabilities.
    """
    # Initialize the grid
    grid = np.zeros((M, M, M))
    
    # Convert fractional coordinates to grid indices
    indices = (li_coords_all * M).astype(int)

    # Ensure indices are within bounds
    indices = np.clip(indices, 0, M-1)
    
    # Count the points in each grid cell
    for index in indices:
        grid[tuple(index)] += 1
    
    # Normalize the grid so that the sum of all values is 1
    total_points = len(li_coords_all)
    grid /= total_points
    
    return grid


def cpsat_core_from_indices(li_indices, mn_indices, N_li, proximity_groups=[]):
    model = cp_model.CpModel()

    # A) sites & options
    site_options, var2siteopt, li_sites, mn_sites = build_site_option_maps_from_indices(
        li_indices, mn_indices
    )

    # B) x vars + one-hot per site
    x = build_x_vars_and_onehot(model, site_options)

    # C) Li constraints
    add_li_mn_charge_balance_constraints(model, x, li_sites, mn_sites, N_li)
    add_li_proximity_exclusions(model, x, proximity_groups)

    # (No objective yet; we’ll add it when we map your pair energies W)
    return model, x, site_options, var2siteopt, li_sites, mn_sites


def find_fractional_centers(M):
    """
    Find the fractional coordinates of the centers of the grid cells.
    
    Parameters:
    M (int): Size of the grid along each dimension.
    
    Returns:
    ndarray: MxMxM array of fractional coordinates of the centers.
    """
    centers = np.zeros((M, M, M, 3))
    for i in range(M):
        for j in range(M):
            for k in range(M):
                centers[i, j, k] = [(i + 0.5) / M, (j + 0.5) / M, (k + 0.5) / M]
    return centers


def find_top_x_points(grid, centers, x):
    """
    Find the top x points in the grid in terms of probability.
    
    Parameters:
    grid (ndarray): MxMxM array of probabilities.
    centers (ndarray): MxMxM array of fractional coordinates of the centers.
    x (int): Number of top points to find.
    
    Returns:
    list: List of tuples (fractional_coordinate, probability) for the top x points.
    """
    # Flatten the grid and the coordinates
    flat_grid = grid.flatten()
    flat_centers = centers.reshape(-1, 3)
    
    # Get the indices of the top x values
    top_indices = np.argsort(flat_grid)[-x:]
    
    # Get the top x values and their corresponding fractional coordinates
    top_points = [(flat_centers[i], flat_grid[i]) for i in top_indices]
    
    # Sort the top points by probability in descending order
    top_points.sort(key=lambda x: x[1], reverse=True)
    
    
    return top_points


def generate_filtered_grid(structure, N_initial_grid=1000, min_dist_grid=1.5):
    lattice = structure.lattice.matrix         # 3x3 array
    cart_coords = structure.cart_coords        # (N_atoms, 3)

    # Estimate the number of points per dimension
    volume = np.abs(np.linalg.det(lattice))
    spacing = (volume / N_initial_grid) ** (1/3)
    
    # Determine the number of grid points along each lattice vector
    lengths = np.linalg.norm(lattice, axis=1)
    num_points = np.maximum(np.round(lengths / spacing).astype(int), 1)
    
    # Create fractional grid
    x = np.linspace(0, 1, num_points[0], endpoint=False)
    y = np.linspace(0, 1, num_points[1], endpoint=False)
    z = np.linspace(0, 1, num_points[2], endpoint=False)
    grid_frac = np.array(np.meshgrid(x, y, z, indexing='ij')).reshape(3, -1).T  # (M, 3)

    # Convert to Cartesian coordinates
    grid_cart = grid_frac @ lattice  # (M, 3)

    # Remove grid points that are too close to any atom
    tree = cKDTree(cart_coords)
    distances, _ = tree.query(grid_cart, k=1)
    mask = distances > min_dist_grid
    filtered_grid = grid_cart[mask]

    return filtered_grid


def init_run_store(
    output_dir: str,
    initial_structure,                 # pymatgen.Structure (framework, no Li)
    li_sites: list,                    # CP site IDs for Li grid
    mn_sites: list,                    # CP site IDs for Mn
    initial_grid_cart: np.ndarray,     # (M,3) Li grid in CARTESIAN
    mn_atom_indices: list,             # len == len(mn_sites); atom indices in initial_structure
    QUBO_ut: np.ndarray,               # upper-triangular QUBO matrix (n x n)
    SCALE: int,                        # integer scaling used in objective
    solver_params: dict,               # e.g. {"time":180,"workers":8,"seed":42}
    extra_meta: dict = None,           # optional extra info
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "energy_model"), exist_ok=True)

    # --- Geometry data ---
    lat = initial_structure.lattice.matrix.astype(np.float32)
    species_Z = np.array([sp.Z for sp in initial_structure.species], dtype=np.int16)
    frac_coords = initial_structure.frac_coords.astype(np.float32)
    li_grid_frac = initial_structure.lattice.get_fractional_coords(initial_grid_cart).astype(np.float32)

    # Save geometry
    np.savez_compressed(
        os.path.join(output_dir, "geometry.npz"),
        lattice=lat,
        species_Z=species_Z,
        frac_coords=frac_coords,
        li_grid_frac=li_grid_frac,
        mn_atom_indices=np.array(mn_atom_indices, dtype=np.int32),
    )

    # Save mapping
    with open(os.path.join(output_dir, "mapping.json"), "w") as f:
        json.dump(
            {"li_sites": list(map(int, li_sites)), "mn_sites": list(map(int, mn_sites))},
            f,
            indent=2,
        )

    # Save QUBO model
    np.savez_compressed(
        os.path.join(output_dir, "energy_model", "qubo_ut.npz"),
        Q_ut=QUBO_ut.astype(np.float32),
        SCALE=int(SCALE),
    )

    # --- Hashes for provenance ---
    geom_hash = _sha256_of_arrays(lat, species_Z, frac_coords, li_grid_frac)
    qubo_hash = _sha256_of_arrays(QUBO_ut)

    # --- Meta info ---
    meta = {
        "run_id": os.path.basename(output_dir),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "solver_params": solver_params,
        "SCALE": int(SCALE),
        "geom_hash": geom_hash,
        "qubo_hash": qubo_hash,
        "files": {
            "geometry": "geometry.npz",
            "mapping": "mapping.json",
            "qubo": "energy_model/qubo_ut.npz",
            "incumbents": "incumbents.jsonl.gz",
        },
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {"geom_hash": geom_hash, "qubo_hash": qubo_hash}


def iter_incumbent_records(output_dir, *, keep_final_only=False, dedup_cfg=True):
    """
    Yields incumbent dicts from incumbents.jsonl.gz.
    - keep_final_only=True: only yield those with tags.status == 'FINAL'
    - dedup_cfg=True: skip duplicates by 'cfg' hash
    """
    inc_path = os.path.join(output_dir, "incumbents.jsonl.gz")
    seen = set()
    with gzip.open(inc_path, "rt") as f:
        for line in f:
            rec = json.loads(line)
            if keep_final_only and not (rec.get("tags", {}).get("status") == "FINAL"):
                continue
            if dedup_cfg:
                cfg = rec.get("cfg")
                if cfg in seen:
                    continue
                seen.add(cfg)
            yield rec


def join_structure_grid(structure,initial_grid):

    initial_grid_pmg = Structure(structure.lattice,[3]*len(initial_grid),initial_grid,coords_are_cartesian=True)
    
    return Structure.from_sites(initial_grid_pmg.sites+structure.sites)


def load_run_assets(output_dir):
    """
    Returns:
      lattice: Lattice
      base_struct: Structure (framework only, no Li; all Mn as Mn element)
      li_sites: list[int]
      mn_sites: list[int]
      li_grid_frac: (M,3) np.ndarray fractional coords aligned with li_sites
      mn_atom_indices: list[int] atom indices in base_struct aligned with mn_sites
    """
    geom = np.load(os.path.join(output_dir, "geometry.npz"))
    lat_mat = geom["lattice"]
    species_Z = geom["species_Z"]
    frac_coords = geom["frac_coords"]
    li_grid_frac = geom["li_grid_frac"]
    mn_atom_indices = geom["mn_atom_indices"].tolist()

    with open(os.path.join(output_dir, "mapping.json"), "r") as f:
        mapping = json.load(f)
    li_sites = mapping["li_sites"]
    mn_sites = mapping["mn_sites"]

    lattice = Lattice(lat_mat)
    species = [Element.from_Z(int(z)) for z in species_Z]
    base_struct = Structure(lattice, species, frac_coords, coords_are_cartesian=False)
    return lattice, base_struct, li_sites, mn_sites, li_grid_frac, mn_atom_indices


def make_output_dir(base="runs", prefix="out"):
    """
    Create a unique output directory under `base/` with a compact timestamp.

    Example: runs/out_20251112_153042_001
    """
    os.makedirs(base, exist_ok=True)
    # UTC timestamp in YYYYMMDD_HHMMSS format
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    # append a short random or incremental suffix to avoid clashes within same second
    suffix = str(int(time.time() * 1000) % 1000).zfill(3)
    name = f"{prefix}_{stamp}_{suffix}"
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)
    return path


def parse_gulp_to_pymatgen(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines[::-1]:
        if 'Total lattice energy ' in line and ' eV ' in line:
            energy = float(line.split()[-2])
            break
 
    for line in lines:
        if 'Total number atoms/shells' in line:
            n_atoms = int(line.strip().split()[-1])
            break
    
    # --- 1. Extract lattice parameters ---
    a = b = c = alpha = beta = gamma = None
    for line in lines:
        if "Final cell parameters and derivatives" in line:
            continue
        if "a =" in line and "alpha" in line:
            parts = line.strip().split()
            a = float(parts[2])
            alpha = float(parts[5])
        elif "b =" in line and "beta" in line:
            parts = line.strip().split()
            b = float(parts[2])
            beta = float(parts[5])
        elif "c =" in line and "gamma" in line:
            parts = line.strip().split()
            c = float(parts[2])
            gamma = float(parts[5])
            break  # we’re done once all 3 are found
 
    if None in (a, b, c, alpha, beta, gamma):
        raise ValueError("Could not parse complete lattice parameters.")

    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

    # --- 2. Extract atomic positions ---
    species = []
    coords = []
    parsing_atoms = False
    for i,line in enumerate(lines):
        if "Final fractional coordinates of atoms" in line:
            parsing_atoms = True
            continue
        if parsing_atoms:
            # if not line.strip():
                # break  # blank line → end of section
            tokens = line.strip().split()
            if len(tokens) > 0:
                if tokens[0] == '1':
                    for j in range(n_atoms):
                        tokens = lines[i+j].strip().split()
                        species.append(tokens[1])

                        coords.append([float(tokens[3]), float(tokens[4]), float(tokens[5])])
                    break
                      

    structure = Structure(lattice, species, coords, coords_are_cartesian=False)
    structure.translate_sites(np.arange(structure.num_sites),[1,1,1],to_unit_cell=True)
    
    return structure, energy


def read_opt_structures(N_structures_opt, folder_path, input_name='gulp_klmc.gin'):

    output_name = input_name[:-3]+'gout'
    all_opt_structures = []
    all_opt_energy = []
    for i in range(N_structures_opt):
        file_path = os.path.join(folder_path,f'A{i}')
        file_path = os.path.join(file_path,output_name)  
        print(file_path)
        if os.path.exists(file_path):
            structure, energy = parse_gulp_to_pymatgen(file_path)
            all_opt_structures.append(structure)
            all_opt_energy.append(energy)

    return all_opt_structures, all_opt_energy


def reduce_qubo_discrete_limno(Q_discrete, species_vector):
    # Convert species to atomic numbers
    atomic_number_vector = np.array([Element(sym).Z for sym in species_vector])

    # Identify indices
    oxygen_positions = np.where(atomic_number_vector == 8)[0]
    other_element_positions = np.where(atomic_number_vector != 8)[0]

    oo_energy = np.sum(Q_discrete[np.ix_(oxygen_positions, oxygen_positions)])

    # Make a copy of the matrix
    Q = copy.deepcopy(Q_discrete)

    # Transfer O-X interaction into X-X diagonal
    for i in other_element_positions:
        for j in oxygen_positions:
            Q[i][i] += Q[i][j]

    # Remove rows and columns corresponding to O atoms
    Q_reduced = np.delete(Q, oxygen_positions, axis=0)
    Q_reduced = np.delete(Q_reduced, oxygen_positions, axis=1)

    return Q_reduced, oo_energy


def structure_from_incumbent_record(
    record: dict,
    base_struct: Structure,
    li_sites: list[int],
    mn_sites: list[int],
    li_grid_frac: np.ndarray,
    mn_atom_indices: list[int],
    *,
    encode_mn_as_tc: bool = True,   # match your GULP script (Mn3 -> 'Tc')
    set_oxidation: bool = False      # alternative: Mn3/4 as oxidation states on Mn
) -> Structure:
    """
    record is a dict from incumbents.jsonl.gz with keys: li_on, mn3_on, ...
    Returns a new Structure with Li added and Mn adjusted (either Tc element or oxidation).
    """
    struct = base_struct.copy()

    # 1) Adjust Mn sites: mn_sites[i] -> atom index mn_atom_indices[i]
    mn3_set = set(record.get("mn3_on", []))
    for i, s in enumerate(mn_sites):
        atom_idx = mn_atom_indices[i]
        if s in mn3_set:
            if encode_mn_as_tc:
                struct.replace(atom_idx, Element("Tc"))  # your pipeline expects Tc for Mn3+
            elif set_oxidation:
                struct.replace(atom_idx, Species("Mn", 3))
        else:
            if set_oxidation:
                struct.replace(atom_idx, Species("Mn", 4))
            # else: leave as elemental Mn

    # 2) Add Li atoms at selected grid sites
    li_pos = {s: i for i, s in enumerate(li_sites)}
    for s in record.get("li_on", []):
        gi = li_pos[s]
        frac = li_grid_frac[gi]
        struct.append("Li", frac, coords_are_cartesian=False)

    return struct


def structure_hash_pbc(pmg_structure, tol_frac=1e-4, tol_lat=1e-3):
    """
    Order- and translation-invariant hash:
      - lattice matrix (rounded)
      - sorted list of (symbol, rounded fractional coords)
    """
    lat = pmg_structure.lattice.matrix
    lat_key = tuple(_round_array(lat, tol_lat).flatten())

    fracs = pmg_structure.frac_coords % 1.0
    fracs_key = _round_array(fracs, tol_frac)
    syms = [str(sp) for sp in pmg_structure.species]
    rows = sorted((syms[i], fracs_key[i,0], fracs_key[i,1], fracs_key[i,2]) for i in range(len(syms)))

    return hash((lat_key, tuple(rows), len(syms)))


def unwrap_frac_coords(frac_coords_list):
    """
    Aligns fractional coordinates modulo lattice to a reference so that they 
    can be meaningfully averaged. Assumes all coords_list[i] are (N_atoms, 3).
    """
    # Use the first set of coordinates as reference
    ref = frac_coords_list[0]
    unwrapped_list = []

    for coords in frac_coords_list:
        delta = coords - ref
        # Apply minimum image convention
        delta -= np.round(delta)
        aligned = ref + delta
        unwrapped_list.append(aligned)

    return np.array(unwrapped_list)


def write_gulp_input(structure, filename="gulp_input.gin"):
    with open(filename, "w") as f:
        f.write("sp opti fbfgs conp #property phon comp\n")
        f.write("vectors\n")
        for line in structure.lattice.matrix:
            f.write(" ".join([f"{x:.6f}" for x in line]) + "\n")
        
        f.write("0 0 0 0 0 0\n")
        f.write("cartesian\n")
        
        for an, line in zip(structure.atomic_numbers, structure.cart_coords):
            symbol = Element.from_Z(an).symbol
            f.write(f"{symbol} core {line[0]:.6f} {line[1]:.6f} {line[2]:.6f}\n")
        
        f.write("\nspecies\n")
        f.write("Mn     core    4.000000\n")
        f.write("Tc     core    3.000000\n")
        f.write("Li     core    1.000000\n")
        f.write("O      core   -2.000000\n")
        
        f.write("buck\n")
        f.write("Li core O core 426.480     0.3000     0.00 0.00 25.00\n")
        f.write("Mn core O core 3087.826    0.2642     0.00 0.00 25.00\n")
        f.write("Tc core O core 1686.125    0.2962     0.00 0.00 25.00\n")
        f.write("O  core O core 22.410      0.6937    32.32 0.000 25.00\n")


def write_gulp_inputs_from_incumbents(
        output_dir: str,
        dest_dir: str,
        *,
        limit: int | None = None,
        keep_final_only: bool = False,
        filename_pattern: str = "A{idx}.gin",
        encode_mn_as_tc: bool = True,
        set_oxidation: bool = False,
        write_gulp_input_fn=None,   # pass your write_gulp_input if not global
        # --- new options for batch files ---
        write_taskfarm: bool = True,
        write_slurm: bool = True,
        job_name: str = "gulp_run",
        account: str = "e05-algor-smw",
        partition: str = "standard",
        qos: str = "short",
        exe_path: str = "/work/e05/e05/bcamino/klmc_exe/klmc3.062024.x",
        ntasks: int = 128,                 # total MPI ranks
        ntasks_per_node: int = 128,        # ranks per node
        cpus_per_task: int = 1,
        dedup_by_hash: bool = True,
    ):
        """
        Rebuild structures for incumbents and write GULP .gin files.
        Also writes:
        - taskfarm.config (task_start 0, task_end N-1)
        - SLURM_js.slurm (submission script)
        """
        from pathlib import Path

       

        os.makedirs(dest_dir, exist_ok=True)
        write_gulp_input = None
        if write_gulp_input_fn is not None:
            write_gulp_input = write_gulp_input_fn
        elif "write_gulp_input" in globals():
            write_gulp_input = globals()["write_gulp_input"]
        elif _default_write_gulp_input is not None:
            write_gulp_input = _default_write_gulp_input
        if write_gulp_input is None:
            raise RuntimeError("write_gulp_input function is not available; pass write_gulp_input_fn or ensure full_script_functions is importable.")
        count = 0
        # load run assets
        lattice, base_struct, li_sites, mn_sites, li_grid_frac, mn_atom_indices = load_run_assets(output_dir)

        # write all incumbents (or FINAL only) as gin files
        for idx, rec in enumerate(iter_incumbent_records(output_dir, keep_final_only=keep_final_only, dedup_cfg=dedup_by_hash)):
            struct = structure_from_incumbent_record(
                rec, base_struct, li_sites, mn_sites, li_grid_frac, mn_atom_indices,
                encode_mn_as_tc=encode_mn_as_tc, set_oxidation=set_oxidation
            )
            name = filename_pattern.format(idx=idx, cfg=rec.get("cfg", f"{idx:05d}"))
            path = Path(dest_dir) / name
            write_gulp_input(structure=struct, filename=str(path))
            count += 1
            if limit is not None and count >= limit:
                break

        # Nothing to schedule -> return early
        if count == 0:
            # still write empty taskfarm if requested (task_end -1 is odd, so skip)
            return 0
        if count<128:
            ntasks = count
            ntasks_per_node = count
        # ----- Write taskfarm.config -----
        if write_taskfarm:
            tf_path = Path(output_dir) / "gulp/taskfarm.config"
            with open(tf_path, "w") as f:
                f.write("task_start 0\n")
                f.write(f"task_end {count-1}\n")
                f.write("cpus_per_worker 1\n")
                f.write("application gulp\n")

        # ----- Write SLURM_js.slurm -----
        if write_slurm:
            slurm_path = Path(output_dir) / "gulp/SLURM_js.slurm"
            actual_ntasks = ntasks if count >= ntasks else count
            script = textwrap.dedent(f"""\
                #!/bin/bash

                #SBATCH --job-name={job_name}
                #SBATCH --time=00:20:00
                #SBATCH --nodes=1
                #SBATCH --account={account}
                #SBATCH --partition={partition}
                #SBATCH --qos={qos}

                export OMP_NUM_THREADS=1

                EXE="{exe_path}"
                srun -n {actual_ntasks} --ntasks-per-node={ntasks_per_node} --cpus-per-task={cpus_per_task} --distribution=block:block --hint=nomultithread --exact ${{EXE}} 1> stdout 2> stderr

                mkdir -p result
                mv A* ./result 2>/dev/null || true

                mkdir -p log
                mv master.log ./log 2>/dev/null || true
                mv workgroup*.log ./log 2>/dev/null || true
                """)
            with open(slurm_path, "w") as f:
                f.write(script)

        return count

def _bron_kerbosch(R, P, X, adj, cliques, max_nodes=1000):
    """Simple Bron–Kerbosch to enumerate maximal cliques (no pivot)."""
    # small safeguard
    if len(cliques) > max_nodes:
        return
    if not P and not X:
        if len(R) >= 2:
            cliques.append(tuple(sorted(R)))
        return
    # iterate over a copy since P will mutate
    for v in list(P):
        _bron_kerbosch(R | {v}, P & adj[v], X & adj[v], adj, cliques, max_nodes)
        P.remove(v)
        X.add(v)


def _buckingham_worker(args):
    (i_start, i_end,
     fcoords, ccoords, sites, lat,
     species_dict, buckingham_params, unique_rhos, R_max,
     distance_analysis, distance_threshold,
     index_map, expanded_N) = args

    B_local = np.zeros((expanded_N, expanded_N), dtype=float)

    # Helper to canonicalize pair keys
    def canon_pair(a, b):
        return (a, b) if a <= b else (b, a)

    for i in range(i_start, i_end):
        ri = ccoords[i]
        spi = str(sites[i].specie)
        opts_i = species_dict.get(spi, [spi])
        inds_i = index_map[i]

        # Get periodic neighbors within cutoff
        nf, dists, js, _ = lat.get_points_in_sphere(
            fcoords, ri, R_max, zip_results=False
        )

        mask = dists > 1e-12
        dists = dists[mask]
        js = js[mask]

        if dists.size == 0:
            continue

        # Group neighbors by j
        js_unique, inv = np.unique(js, return_inverse=True)

        for group_idx, j in enumerate(js_unique):
            if j <= i:
                continue

            block_mask = (inv == group_idx)
            dj = dists[block_mask]
            if dj.size == 0:
                continue

            # Safety check
            if distance_analysis and dj.min() < distance_threshold:
                for ii in inds_i:
                    for jj in index_map[j]:
                        B_local[ii, jj] += 1e6
                        B_local[jj, ii] += 1e6
                continue

            # Per-(i,j) vectorized terms
            inv_r6 = (dj**-6).sum()
            Sexp = {rho: np.exp(-dj / rho).sum() for rho in unique_rhos}

            spj = str(sites[j].specie)
            opts_j = species_dict.get(spj, [spj])
            inds_j = index_map[j]

            for ii, ei in zip(inds_i, opts_i):
                for jj, ej in zip(inds_j, opts_j):
                    key = canon_pair(ei, ej)
                    if key not in buckingham_params:
                        continue
                    A, rho, C = buckingham_params[key]

                    Vij = A * Sexp[rho] - C * inv_r6
                    B_local[ii, jj] += Vij
                    B_local[jj, ii] += Vij

    return B_local


def _frac_coords(coords_cart, lattice):
    """Convert cartesian coords to fractional with given 3x3 lattice matrix."""
    return np.dot(coords_cart, np.linalg.inv(lattice).T)


def _maximal_cliques_from_edges(N, edges, cap_cliques=10000):
    """Return list of maximal cliques (each a tuple of node indices)."""
    # adjacency as sets
    adj = [set() for _ in range(N)]
    for i, j in edges:
        adj[i].add(j); adj[j].add(i)
    cliques = []
    _bron_kerbosch(set(), set(range(N)), set(), adj, cliques, max_nodes=cap_cliques)
    return cliques


def _mic_delta_frac(df):
    """Minimum-image wrap for fractional deltas to (-0.5, 0.5]."""
    return df - np.round(df)


def _pair_edges_with_threshold(frac_coords, lattice, threshold):
    """
    Build edges (i,j) where PBC distance < threshold.
    frac_coords: (N,3) fractional coords in [0,1)
    lattice: 3x3 cartesian lattice matrix
    """
    N = len(frac_coords)
    edges = []
    L = np.asarray(lattice)
    for i in range(N):
        for j in range(i+1, N):
            df = _mic_delta_frac(frac_coords[j] - frac_coords[i])
            dcart = df @ L
            if np.linalg.norm(dcart) < threshold:
                edges.append((i, j))
    return edges


def _round_array(a, tol):
    return np.round(a / tol).astype(np.int64)


def _sha256_of_arrays(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]  # short hash for filenames/IDs







# -------------------------------------------------------------------
# MAIN PARALLEL FUNCTION
# -------------------------------------------------------------------


# THE PARAM

N_li = 2

N_initial_grid=100
min_dist_grid=1.

threshold_li=1.5
prox_penalty=1000

one_hot_value = 200
weight = 500

N_structures_opt = 2

number_iterations = 1000
number_runs = 100

input_name='gulp_klmc.gin'
gulp_io_path='klmc/'
mace_io_path='mace_io_files'

M = 20 #grid definition
N_positions_final = 100

threshold = 0.1  # THIS IS AN IMPORTANT PARAMETER

num_iterations = 5

max_time = 60

DEDUP_INCUMBENTS = False



# THE LOOP
initial_structure = Structure.from_file('delithiated_tmp.cif')

initial_grid = generate_filtered_grid(
    initial_structure,
    N_initial_grid=N_initial_grid,
    min_dist_grid=min_dist_grid
)
structure = join_structure_grid(initial_structure, initial_grid)
grid = copy.deepcopy(initial_grid)

folder_path = os.path.abspath(make_output_dir(base="runs", prefix="out"))
for i in range(num_iterations):
    print(f'************ Begin Iteration {i} ************')

    output_dir = os.path.join(folder_path,f'output_folder_{i}')

    # --- Build energy matrix (QUBO) and mapping ---
    QUBO, li_indices, mn_indices = build_QUBO(
        structure,
        threshold_li=threshold_li,
        prox_penalty=prox_penalty
    )

    # CP-SAT core (one-hot, counts, charge balance)
    model, x, site_options, var2siteopt, li_sites, mn_sites = cpsat_core_from_indices(
        li_indices, mn_indices, N_li=N_li
    )

    groups, pairs = build_li_proximity_groups(
        li_grid_coords=grid,               # (M,3) Cartesian
        threshold_ang=1.8,                 # your cutoff
        lattice=initial_structure.lattice, # for PBC-aware distances
        coords_are_cartesian=True,
        site_ids=li_sites                  # CP site IDs aligned to grid order
    )
    add_li_proximity_exclusions(model, x, groups)

    # Objective from upper-triangular QUBO
    SCALE, info = add_ut_qubo_objective(model, x, var2siteopt, QUBO)
    print("Objective wiring:", info)

    # --- Mn atom indices (Z=25). Align to mn_sites order if needed ---
    mn_atom_indices_all = np.where(np.array(initial_structure.atomic_numbers) == 25)[0]
    assert len(mn_atom_indices_all) >= len(mn_sites), \
        "Not enough Mn atoms in initial_structure to map mn_sites."
    mn_atom_indices = list(mn_atom_indices_all[:len(mn_sites)])
    n_workers = multiprocessing.cpu_count()
    # --- Save run-level artifacts once ---
    solver_params = {"time": max_time, "workers": n_workers, "seed": 2025020}
    _ = init_run_store(
        output_dir=output_dir,
        initial_structure=initial_structure,
        li_sites=li_sites,
        mn_sites=mn_sites,
        initial_grid_cart=grid,
        mn_atom_indices=mn_atom_indices,
        QUBO_ut=QUBO,
        SCALE=SCALE,
        solver_params=solver_params,
    )

    # --- Solver setup ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time
    solver.parameters.num_search_workers = n_workers
    solver.parameters.random_seed = solver_params["seed"]
    solver.parameters.log_search_progress = True
    if hasattr(solver.parameters, "use_lns"):
        solver.parameters.use_lns = True

    # --- Create and attach incumbent saver callback ---
    cb = StreamingIncumbentSaver(
        x=x,
        site_options=site_options,
        li_sites=li_sites,
        mn_sites=mn_sites,
        scale=SCALE,
        out_dir=output_dir,
        limit=500,  # optional cap on saved incumbents
    )

    # --- Solve with callback ---
    status = solver.Solve(model, cb)
    print("Status:", solver.StatusName(status))
    print("Incumbents saved during search:", cb.count)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Decode final best assignment
        assignment = {s: next(a for a in opts if solver.Value(x[(s, a)]) == 1)
                      for s, opts in site_options.items()}
        try:
            best_E = solver.ObjectiveValue() / SCALE
        except Exception:
            best_E = None

        # Save terminal incumbent (tagged as FINAL)
        cfg_hash = append_incumbent(
            output_dir=output_dir,
            assignment=assignment,
            energy_ev=best_E,
            li_sites=li_sites,
            mn_sites=mn_sites,
            tags={"status": "FINAL", "solver_status": solver.StatusName(status)}
        )
        print(f"Saved final incumbent cfg: {cfg_hash}, E = {best_E:.3f} eV")
    else:
        print("No feasible solution found.")

    # 2) Or dump the first 50 distinct incumbents found during search
    n = write_gulp_inputs_from_incumbents(
        output_dir=output_dir,
        dest_dir=os.path.join(output_dir, "gulp/run"),
        limit=50,
        keep_final_only=False,
        filename_pattern="A{idx}.gin",
        write_gulp_input_fn=write_gulp_input,
        dedup_by_hash=DEDUP_INCUMBENTS,
    )
    print("Incumbents dumped:", n)

    bash_script = os.path.join(output_dir, 'gulp', 'SLURM_js.slurm')
    gulp_cwd = os.path.join(output_dir, 'gulp')
    print(f"About to run SLURM script: {bash_script}")
    if not os.path.exists(bash_script):
        print(f"SLURM script not found at {bash_script}. Skipping execution and continuing.")
        continue
    try:
        result = subprocess.run(
            ["bash", bash_script],
            cwd=gulp_cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            print("SLURM script STDOUT:\n", result.stdout)
        if result.stderr:
            print("SLURM script STDERR:\n", result.stderr)
    except subprocess.CalledProcessError as exc:
        print(f"SLURM script failed with return code {exc.returncode}")
        if exc.stdout:
            print("Captured STDOUT:\n", exc.stdout)
        if exc.stderr:
            print("Captured STDERR:\n", exc.stderr)
        raise

    opt_structures, all_opt_energy = read_opt_structures(N_structures_opt, os.path.join(output_dir,'gulp/result'), input_name)
    print(f'Iteration {i} - GULP opt_structures = {len(opt_structures)}')

    structure, grid_frac = build_new_structural_model(
        opt_structures, M, N_positions_final, initial_structure, threshold, return_stats = False, fix_angles=True)
    grid = structure.lattice.get_cartesian_coords(grid_frac)
    # folder_path = "runs/out_20251112_151804_526"
    extxyz_path = f"{folder_path}/all_optimized.extxyz"  # one cumulative file

    # After you get opt_structures, all_opt_energy for an iteration:
    added, skipped_old, skipped_dup = append_unique_extxyz(
        extxyz_path,
        structures=opt_structures,
        energies=all_opt_energy,
        tol_frac=1e-4,   # ~1e-4 of fractional coordinate ~ good default
        tol_lat=1e-3     # 0.001 Å lattice tolerance
    )

    print(f"extxyz update → added: {added}, skipped(existing): {skipped_old}, skipped(batch-dup): {skipped_dup}")

    print(f'************ End Iteration {i} ************\n')
