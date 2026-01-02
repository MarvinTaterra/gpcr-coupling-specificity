"""MDPath --- MD signal transduction calculation and visualization --- :mod:`mdpath.mdapth`
====================================================================

MDPath is a Python package for calculating signal transduction paths in molecular dynamics (MD) simulations. 
The package uses mutual information to identify connections between residue movements.
Using a graph shortest paths with the highest mutual information are calculated.
Paths are then clustered based on the overlap between them to identify a continuous network throught the analysed protein.
The package also includes functionalitys for the visualization of results.

Release under the MIT License. See LICENSE for details.

This is the main script of MDPath. It is used to run MDPath from the command line.
MDPath can be called from the comadline using 'mdapth' after instalation
Use the -h flag to see the available options.
"""

import os
import argparse
import pandas as pd
import numpy as np
import MDAnalysis as mda
import json
import multiprocessing
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import pickle
from joblib import dump, load

from mdpath.src.structure import StructureCalculations, DihedralAngles
from mdpath.src.mutual_information import NMICalculator
from mdpath.src.graph import GraphBuilder
from mdpath.src.cluster import PatwayClustering
from mdpath.src.visualization import MDPathVisualize
from mdpath.src.bootstrap import BootstrapAnalysis
import os


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def convert_pathways_to_native_int(pathways):
    """Convert all residue numbers in pathways to native Python int.
    
    This ensures consistent dictionary lookups regardless of whether
    the original values were np.int64 or Python int.
    
    Args:
        pathways: List of pathways, where each pathway is a list of residue numbers
        
    Returns:
        List of pathways with all residue numbers as native Python int
    """
    converted = []
    for pathway in pathways:
        converted_pathway = [int(res) for res in pathway]
        converted.append(converted_pathway)
    return converted


def convert_cluster_dict_to_native_int(cluster_dict):
    """Convert all residue numbers in cluster pathways dict to native Python int.
    
    Args:
        cluster_dict: Dictionary mapping cluster IDs to lists of pathways
        
    Returns:
        Dictionary with all residue numbers as native Python int
    """
    converted = {}
    for cluster_id, pathways in cluster_dict.items():
        converted[cluster_id] = convert_pathways_to_native_int(pathways)
    return converted


def renumber_residues_from_one(universe):
    """
    Renumber all residues in the universe starting from 1.
    
    Args:
        universe: MDAnalysis Universe object
    
    Returns:
        Modified universe with renumbered residues
    """
    # Get unique residues in order
    residues = universe.residues
    
    # Create mapping from old to new residue numbers
    resnum_map = {}
    for new_num, residue in enumerate(residues, start=1):
        old_num = residue.resid
        resnum_map[old_num] = new_num
        # Update residue number
        residue.resid = new_num
    
    print(f"  Renumbered {len(residues)} residues (1 to {len(residues)})")
    return universe


def main():
    """Main function for running MDPath from the command line.
    It can be called using 'mdpath' after installation.

    Command-line inputs:
        -top: Topology file of your MD simulation

        -traj: Trajectory file of your MD simulation

        -cpu: Amount of cores used in multiprocessing (default: half of available cores)

        -lig: Protein ligand interacting residues (default: False)

        -bs: How often bootstrapping should be performed (default: False)

        -fardist: Default distance for faraway residues (default: 12.0)

        -closedist: Default distance for close residues (default: 12.0)

        -graphdist: Default distance for residues making up the graph (default: 5.0)

        -numpath: Default number of top paths considered for clustering (default: 500)
        
        -chain: Chain of the protein to be analyzed
        
        -segid: Segment ID to be analyzed (e.g., PROA, PROB) with automatic renumbering from 1
    """
    parser = argparse.ArgumentParser(
        prog="mdpath",
        description="Calculate signal transduction paths in your MD trajectories",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-top",
        dest="topology",
        help="Topology file of your MD simulation",
        required=False,
    )
    parser.add_argument(
        "-traj",
        dest="trajectory",
        help="Trajectory file of your MD simulation",
        required=False,
    )
    parser.add_argument(
        "-cpu",
        dest="num_parallel_processes",
        help="Amount of cores used in multiprocessing",
        default=(os.cpu_count() // 2),
    )
    parser.add_argument(
        "-lig",
        dest="lig_interaction",
        help="Protein ligand interacting residues",
        default=False,
        nargs="+",
    )
    parser.add_argument(
        "-bs",
        dest="bootstrap",
        help="How often bootstrapping should be performed.",
        default=False,
    )
    parser.add_argument(
        "-spline",
        dest="spline",
        help="Create STL files for spline visualization.",
        default=True,
    )
    # TODO maybe move settingsflags to a conffile that can be changed
    # Settings Flags
    parser.add_argument(
        "-fardist",
        dest="fardist",
        help="Default distance for faraway residues.",
        required=False,
        default=12.0,
    )
    parser.add_argument(
        "-closedist",
        dest="closedist",
        help="Default distance for close residues.",
        required=False,
        default=12.0,
    )
    parser.add_argument(
        "-graphdist",
        dest="graphdist",
        help="Default distance for residues making up the graph.",
        required=False,
        default=5.0,
    )
    parser.add_argument(
        "-numpath",
        dest="numpath",
        help="Default number of top paths considered for clustering.",
        required=False,
        default=500,
    )

    parser.add_argument(
        "-chain",
        dest="chain",
        help="Chain of the protein to be analyzed in the topology file. CAUTION: only one chain can be selected for analysis.",
        required=False,
        default=False,
    )

    parser.add_argument(
        "-segid",
        dest="segid",
        help="Segment ID (e.g., PROA, PROB) to be analyzed. Residues will be automatically renumbered starting from 1. CAUTION: only one segment can be selected.",
        required=False,
        default=False,
    )

    parser.add_argument(
        "-invert",
        dest="invert",
        help="Inverts NMI bei subtrackting each NMI from max NMI. Can be used to find Paths, that are the least correlated",
        required=False,
        default=False,
    )

    args = parser.parse_args()
    if not args.topology or not args.trajectory:
        print("Both trajectory and topology files are required!")
        exit()

    num_parallel_processes = int(args.num_parallel_processes)
    topology = args.topology
    trajectory = args.trajectory
    traj = mda.Universe(topology, trajectory)
    lig_interaction = args.lig_interaction
    bootstrap = args.bootstrap
    fardist = float(args.fardist)
    closedist = float(args.closedist)
    graphdist = float(args.graphdist)
    numpath = int(args.numpath)
    invert = bool(args.invert)
    spline = bool(args.spline)

    # Segment ID selection (takes priority over chain selection)
    if args.segid:
        segid = str(args.segid)
        print(f"\n{'='*70}")
        print(f"SEGMENT ID SELECTION: {segid}")
        print(f"{'='*70}")
        
        # Select atoms by segment ID
        segid_atoms = traj.select_atoms(f"segid {segid}")
        if len(segid_atoms) == 0:
            raise ValueError(f"No atoms found for segment ID {segid}")
            exit()
        
        print(f"Found {segid_atoms.n_atoms} atoms in segment {segid}")
        print(f"Found {len(segid_atoms.residues)} residues")
        
        # Create new universe with selected atoms
        segid_universe = mda.Merge(segid_atoms)
        
        # Renumber residues starting from 1
        print("Renumbering residues starting from 1...")
        segid_universe = renumber_residues_from_one(segid_universe)
        
        # Write new topology
        segid_universe.atoms.write("selected_segid.pdb")
        print(f"  Written topology to: selected_segid.pdb")
        
        # Write trajectory with renumbered residues
        print("Writing trajectory...")
        with mda.Writer("selected_segid.dcd", segid_atoms.n_atoms) as W:
            for i, ts in enumerate(traj.trajectory):
                segid_universe.atoms.positions = segid_atoms.positions
                W.write(segid_universe.atoms)
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i+1} frames...", end='\r')
        
        print(f"\n  Written trajectory to: selected_segid.dcd")
        
        # Update paths for analysis
        topology = "selected_segid.pdb"
        trajectory = "selected_segid.dcd"
        traj = mda.Universe(topology, trajectory)
        
        print(f"\n✓ Segment {segid} selected and renumbered (residues 1-{len(segid_universe.residues)})")
        print(f"  Analysis will proceed with selected_segid.pdb and selected_segid.dcd")
        print(f"{'='*70}\n")
    
    # Chain selection (only if segid not specified)
    elif args.chain:
        chain = str(args.chain)
        chain_atoms = traj.select_atoms(f"chainID {chain}")
        if len(chain_atoms) == 0:
            raise ValueError(f"No atoms found for chain {chain}")
            exit()
        chain_universe = mda.Merge(chain_atoms)
        
        # Renumber residues for chain selection too
        print("Renumbering residues starting from 1...")
        chain_universe = renumber_residues_from_one(chain_universe)
        
        # Write new topology
        chain_universe.atoms.write("selected_chain.pdb")

        # Write trajectory
        with mda.Writer("selected_chain.dcd", chain_atoms.n_atoms) as W:
            for ts in traj.trajectory:
                chain_universe.atoms.positions = chain_atoms.positions
                W.write(chain_universe.atoms)
        topology = "selected_chain.pdb"
        trajectory = "selected_chain.dcd"
        traj = mda.Universe(topology, trajectory)
        print(
            f"Chain {chain} selected, renumbered, and written to selected_chain.pdb and selected_chain.dcd"
        )

    # Write first_frame.pdb AFTER any segid/chain selection and renumbering
    # This ensures first_frame.pdb has the same numbering as the analysis
    if os.path.exists("first_frame.pdb"):
        os.remove("first_frame.pdb")
    with mda.Writer("first_frame.pdb", multiframe=False) as pdb:
        traj.trajectory[0]
        pdb.write(traj.atoms)

    structure_calc = StructureCalculations(topology)
    df_distant_residues = structure_calc.calculate_residue_suroundings(fardist, "far")
    df_close_res = structure_calc.calculate_residue_suroundings(closedist, "close")
    dihedral_calc = DihedralAngles(
        traj,
        structure_calc.first_res_num,
        structure_calc.last_res_num,
        structure_calc.last_res_num,
    )
    df_all_residues = dihedral_calc.calculate_dihedral_movement_parallel(
        num_parallel_processes
    )
    print("\033[1mTrajectory is processed and ready for analysis.\033[0m")

    # Calculate the mutual information and build the graph
    nmi_calc = NMICalculator(df_all_residues, invert=invert)
    nmi_calc.entropy_df.to_csv("entropy_df.csv", index=False)
    nmi_calc.nmi_df.to_csv("nmi_df.csv", index=False)
    graph_builder = GraphBuilder(
        topology, structure_calc.last_res_num, nmi_calc.nmi_df, graphdist
    )

    # Save the graph for future data science
    dump(graph_builder.graph, "graph.joblib", compress=3)

    MDPathVisualize.visualise_graph(
        graph_builder.graph
    )  # Exports image of the Graph to PNG

    # Calculate paths
    path_total_weights = graph_builder.collect_path_total_weights(df_distant_residues)
    sorted_paths = sorted(path_total_weights, key=lambda x: x[1], reverse=True)
    with open("output.txt", "w") as file:
        for path, total_weight in sorted_paths[:numpath]:
            file.write(f"Path: {path}, Total Weight: {total_weight}\n")
    top_pathways = [path for path, _ in sorted_paths[:numpath]]
    
    # Convert to native Python int to ensure consistent dictionary lookups
    top_pathways = convert_pathways_to_native_int(top_pathways)

    # Calculate the paths including ligand interacting residues
    if lig_interaction:
        try:
            lig_interaction = [int(res) for res in lig_interaction]
        except ValueError:
            print("Error: All -lig inputs must be integers.")
        sorted_paths = [
            path
            for path in sorted_paths
            if any(residue in lig_interaction for residue in path[0])
        ]
        top_pathways = [path for path, _ in sorted_paths[:numpath]]
        # Convert to native Python int
        top_pathways = convert_pathways_to_native_int(top_pathways)
        print("\033[1mLigand pathways gathered..\033[0m")

    # Bootstrap analysis
    if bootstrap:
        num_bootstrap_samples = int(bootstrap)
        bootstrap_analysis = BootstrapAnalysis(
            df_all_residues,
            df_distant_residues,
            sorted_paths,
            num_bootstrap_samples,
            numpath,
            topology,
            structure_calc.last_res_num,
            graphdist,
        )
        file_name = "path_confidence_intervals.txt"
        bootstrap_analysis.bootstrap_write(file_name)
        print(f"Path confidence intervals have been saved to {file_name}")

    # Cluster pathways to get signaltransduction paths
    clustering = PatwayClustering(df_close_res, top_pathways, num_parallel_processes)
    clusters = clustering.pathways_cluster()
    cluster_pathways_dict = clustering.pathway_clusters_dictionary(
        clusters, sorted_paths
    )
    
    # Convert cluster_pathways_dict to native Python int
    cluster_pathways_dict = convert_cluster_dict_to_native_int(cluster_pathways_dict)
    
    # FIX: Use 'topology' instead of hardcoded "first_frame.pdb"
    residue_coordinates_dict = MDPathVisualize.residue_CA_coordinates(
        topology, structure_calc.last_res_num
    )

    # Export residue coordinates and pathways dict for comparisson functionality
    with open("residue_coordinates.pkl", "wb") as pkl_file:
        pickle.dump(residue_coordinates_dict, pkl_file)

    with open("cluster_pathways_dict.pkl", "wb") as pkl_file:
        pickle.dump(cluster_pathways_dict, pkl_file)

    with open("top_pathways.pkl", "wb") as pkl_file:
        pickle.dump(top_pathways, pkl_file)

    # Export the cluster pathways for visualization
    updated_dict = MDPathVisualize.apply_backtracking(
        cluster_pathways_dict, residue_coordinates_dict
    )
    formated_dict = MDPathVisualize.format_dict(updated_dict)
    
    # Write clusters_paths.json (convert numpy types only at serialization)
    with open("clusters_paths.json", "w") as json_file:
        json.dump(convert_numpy_types(formated_dict), json_file, cls=NumpyEncoder)
    
    # Process path properties (use original data structure)
    path_properties = MDPathVisualize.precompute_path_properties(formated_dict)
    
    # Write precomputed_clusters_paths.json (convert numpy types only at serialization)
    with open("precomputed_clusters_paths.json", "w") as out_file:
        json.dump(convert_numpy_types(path_properties), out_file, indent=4, cls=NumpyEncoder)
    
    # Process quick path properties (use original data structure)
    quick_path_properties = MDPathVisualize.precompute_cluster_properties_quick(
        formated_dict
    )
    
    # Write quick_precomputed_clusters_paths.json (convert numpy types only at serialization)
    with open("quick_precomputed_clusters_paths.json", "w") as out_file2:
        json.dump(convert_numpy_types(quick_path_properties), out_file2, indent=4, cls=NumpyEncoder)

    if spline:
        MDPathVisualize.create_splines("quick_precomputed_clusters_paths.json")


if __name__ == "__main__":
    main()