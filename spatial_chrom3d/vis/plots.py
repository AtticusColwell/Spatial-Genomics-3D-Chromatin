import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from ..core.data_structures import TADomain, ChromatinLoop

def plot_contact_matrix(matrix: np.ndarray, title: str = "Hi-C Contact Matrix",
                       log_scale: bool = True, cmap: str = "Reds") -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plot_matrix = np.log10(matrix + 1) if log_scale else matrix
    
    im = ax.imshow(plot_matrix, cmap=cmap, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel("Genomic Position (bins)")
    ax.set_ylabel("Genomic Position (bins)")
    
    plt.colorbar(im, ax=ax, label="Log10(Contact Frequency)" if log_scale else "Contact Frequency")
    
    return fig

def plot_tads_overlay(matrix: np.ndarray, tads: List[TADomain], 
                     resolution: int = 10000) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(np.log10(matrix + 1), cmap="Reds", interpolation='nearest')
    
    for tad in tads:
        start_bin = tad.start // resolution
        end_bin = tad.end // resolution
        
        ax.plot([start_bin, end_bin, end_bin, start_bin, start_bin],
                [start_bin, start_bin, end_bin, end_bin, start_bin],
                'b-', linewidth=2, alpha=0.7)
    
    ax.set_title("TADs overlaid on Hi-C Matrix")
    ax.set_xlabel("Genomic Position (bins)")
    ax.set_ylabel("Genomic Position (bins)")
    plt.colorbar(im, ax=ax, label="Log10(Contact Frequency)")
    
    return fig

def plot_loops_overlay(matrix: np.ndarray, loops: List[ChromatinLoop],
                      resolution: int = 10000) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(np.log10(matrix + 1), cmap="Reds", interpolation='nearest')
    
    for loop in loops:
        anchor1_bin = loop.anchor1_start // resolution
        anchor2_bin = loop.anchor2_start // resolution
        
        circle = plt.Circle((anchor1_bin, anchor2_bin), radius=2, 
                          color='blue', fill=False, linewidth=2)
        ax.add_patch(circle)
        
        circle = plt.Circle((anchor2_bin, anchor1_bin), radius=2,
                          color='blue', fill=False, linewidth=2)
        ax.add_patch(circle)
    
    ax.set_title("Chromatin Loops overlaid on Hi-C Matrix")
    ax.set_xlabel("Genomic Position (bins)")
    ax.set_ylabel("Genomic Position (bins)")
    plt.colorbar(im, ax=ax, label="Log10(Contact Frequency)")
    
    return fig

def plot_insulation_score(insulation_scores: np.ndarray, boundaries: List[int],
                         resolution: int = 10000) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(15, 4))
    
    positions = np.arange(len(insulation_scores)) * resolution
    ax.plot(positions, insulation_scores, 'k-', linewidth=1)
    
    for boundary in boundaries:
        ax.axvline(x=boundary * resolution, color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlabel("Genomic Position (bp)")
    ax.set_ylabel("Insulation Score")
    ax.set_title("TAD Boundary Detection via Insulation Score")
    
    return fig

def plot_contact_decay(matrix: np.ndarray, max_distance: int = 100) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n = matrix.shape[0]
    distances = []
    contact_means = []
    
    for d in range(1, min(max_distance, n)):
        diagonal_contacts = [matrix[i, i+d] for i in range(n-d) if matrix[i, i+d] > 0]
        if diagonal_contacts:
            distances.append(d)
            contact_means.append(np.mean(diagonal_contacts))
    
    ax.loglog(distances, contact_means, 'o-', markersize=4)
    ax.set_xlabel("Genomic Distance (bins)")
    ax.set_ylabel("Mean Contact Frequency")
    ax.set_title("Contact Frequency vs Genomic Distance")
    ax.grid(True, alpha=0.3)
    
    return fig