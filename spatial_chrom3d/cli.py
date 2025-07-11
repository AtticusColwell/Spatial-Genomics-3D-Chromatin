import click
import numpy as np
from pathlib import Path
from .io.loaders import load_hic_matrix, load_bed_file
from .processing.contact_matrix import normalize_contact_matrix
from .analysis.topology import detect_tads, detect_loops
from .vis.plots import plot_contact_matrix, plot_tads_overlay

@click.group()
@click.version_option()
def main():
    """SpatialChrom3D: Advanced toolkit for spatial genomics and 3D chromatin analysis."""
    pass

@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--resolution', '-r', default=10000, help='Resolution in bp')
@click.option('--normalize/--no-normalize', default=True, help='Apply ICE normalization')
def analyze(input_file, output, resolution, normalize):
    """Analyze Hi-C data for TADs and loops."""
    
    click.echo(f"Loading Hi-C data from {input_file}")
    
    if input_file.endswith('.cool'):
        from .io.loaders import load_hic_cooler
        data = load_hic_cooler(input_file)
        matrix = data.contact_matrix
    else:
        matrix = load_hic_matrix(input_file)
    
    if normalize:
        click.echo("Applying ICE normalization...")
        matrix = normalize_contact_matrix(matrix, method='ice')
    
    click.echo("Detecting TADs...")
    tads = detect_tads(matrix, resolution=resolution)
    
    click.echo("Detecting chromatin loops...")
    loops = detect_loops(matrix, resolution=resolution)
    
    click.echo(f"Found {len(tads)} TADs and {len(loops)} loops")
    
    if output:
        output_dir = Path(output)
        output_dir.mkdir(exist_ok=True)
        
        fig = plot_contact_matrix(matrix)
        fig.savefig(output_dir / 'contact_matrix.png', dpi=300, bbox_inches='tight')
        
        fig = plot_tads_overlay(matrix, tads, resolution)
        fig.savefig(output_dir / 'tads_overlay.png', dpi=300, bbox_inches='tight')
        
        click.echo(f"Results saved to {output_dir}")

@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file')
@click.option('--method', default='ice', help='Normalization method')
def normalize(input_file, output, method):
    """Normalize Hi-C contact matrix."""
    
    click.echo(f"Loading and normalizing {input_file}")
    
    matrix = load_hic_matrix(input_file)
    normalized = normalize_contact_matrix(matrix, method=method)
    
    if output:
        np.savetxt(output, normalized, delimiter='\t')
        click.echo(f"Normalized matrix saved to {output}")

@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--resolution', '-r', default=10000, help='Resolution in bp')
@click.option('--min-size', default=3, help='Minimum TAD size in bins')
@click.option('--output', '-o', type=click.Path(), help='Output BED file')
def find_tads(input_file, resolution, min_size, output):
    """Find TAD boundaries in Hi-C data."""
    
    matrix = load_hic_matrix(input_file)
    tads = detect_tads(matrix, resolution=resolution, min_size=min_size)
    
    click.echo(f"Found {len(tads)} TADs")
    
    if output:
        with open(output, 'w') as f:
            for tad in tads:
                f.write(f"{tad.chromosome}\t{tad.start}\t{tad.end}\n")
        click.echo(f"TADs saved to {output}")

if __name__ == '__main__':
    main()