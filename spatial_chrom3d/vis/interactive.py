import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any
from ..core.data_structures import TADomain, ChromatinLoop

def plot_3d_structure(coordinates: np.ndarray, tads: List[TADomain] = None,
                     loops: List[ChromatinLoop] = None) -> go.Figure:
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=coordinates[:, 0],
        y=coordinates[:, 1],
        z=coordinates[:, 2],
        mode='markers+lines',
        marker=dict(size=3, color='blue'),
        line=dict(width=2, color='blue'),
        name='Chromatin Fiber'
    ))
    
    if tads:
        for i, tad in enumerate(tads):
            start_idx = tad.start // 10000
            end_idx = tad.end // 10000
            if end_idx < len(coordinates):
                fig.add_trace(go.Scatter3d(
                    x=coordinates[start_idx:end_idx, 0],
                    y=coordinates[start_idx:end_idx, 1],
                    z=coordinates[start_idx:end_idx, 2],
                    mode='markers',
                    marker=dict(size=5, color='red'),
                    name=f'TAD {i+1}'
                ))
    
    if loops:
        for i, loop in enumerate(loops):
            anchor1_idx = loop.anchor1_start // 10000
            anchor2_idx = loop.anchor2_start // 10000
            if anchor1_idx < len(coordinates) and anchor2_idx < len(coordinates):
                fig.add_trace(go.Scatter3d(
                    x=[coordinates[anchor1_idx, 0], coordinates[anchor2_idx, 0]],
                    y=[coordinates[anchor1_idx, 1], coordinates[anchor2_idx, 1]],
                    z=[coordinates[anchor1_idx, 2], coordinates[anchor2_idx, 2]],
                    mode='lines',
                    line=dict(width=4, color='green'),
                    name=f'Loop {i+1}'
                ))
    
    fig.update_layout(
        title="3D Chromatin Structure",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )
    
    return fig

def plot_interactive_heatmap(matrix: np.ndarray, title: str = "Interactive Hi-C Heatmap") -> go.Figure:
    
    fig = go.Figure(data=go.Heatmap(
        z=np.log10(matrix + 1),
        colorscale='Reds',
        colorbar=dict(title="Log10(Contact Frequency)")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Genomic Position (bins)",
        yaxis_title="Genomic Position (bins)"
    )
    
    return fig

def plot_multi_scale_analysis(matrices: Dict[str, np.ndarray]) -> go.Figure:
    
    n_matrices = len(matrices)
    fig = make_subplots(
        rows=1, cols=n_matrices,
        subplot_titles=list(matrices.keys()),
        specs=[[{"type": "heatmap"}] * n_matrices]
    )
    
    for i, (name, matrix) in enumerate(matrices.items()):
        fig.add_trace(
            go.Heatmap(
                z=np.log10(matrix + 1),
                colorscale='Reds',
                showscale=(i == 0)
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(title="Multi-Scale Hi-C Analysis")
    
    return fig