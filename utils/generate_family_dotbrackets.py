import os
import sys
from pathlib import Path

# Try to import RNA (ViennaRNA)
try:
    import RNA
except ImportError:
    print("Error: ViennaRNA Python bindings not found. Please install ViennaRNA package.")
    print("Example (conda): conda install -c bioconda viennarna")
    sys.exit(1)

def parse_fasta(fasta_file):
    """
    Generator for FASTA file parsing.
    Yields (header, sequence) tuples.
    """
    header = None
    sequence_parts = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                if header:
                    yield header, "".join(sequence_parts)
                header = line[1:]  # Remove '>'
                sequence_parts = []
            else:
                sequence_parts.append(line)
        
        if header:
            yield header, "".join(sequence_parts)

def process_file(input_path, output_path):
    print(f"Processing {input_path} -> {output_path}")
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        return

    count = 0
    with open(output_path, 'w') as out:
        for header, seq in parse_fasta(input_path):
            # Clean sequence
            seq_upper = seq.upper().replace('T', 'U')
            
            # Calculate MFE structure
            # RNA.fold returns (structure, mfe)
            structure, mfe = RNA.fold(seq_upper)
            
            # Write to output
            # Only the dot-bracket structure string
            # structure
            # Split structure and MFE if needed, here we just output structure
            # structure, mfe = RNA.fold(seq_upper)
            # structure string usually contains the structure and maybe energy in some formats
            # But RNA.fold returns the structure string and the float energy.
            # We just want the structure string.
            
            out.write(f"{structure}\n")
            
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} sequences...", end='\r')
            
    print(f"Finished processing {count} sequences for {input_path.name}.")

def main():
    # Base directory
    base_dir = Path(__file__).resolve().parents[1]
    
    # Data directory
    data_dir = base_dir / "data" / "ft" / "rna_families"
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist.")
        sys.exit(1)
        
    files = ["train_ft", "val_ft", "test_ft"]
    
    for name in files:
        input_file = data_dir / f"{name}.fasta"
        output_file = data_dir / f"{name}_dotbracket.txt"
        
        process_file(input_file, output_file)

if __name__ == "__main__":
    main()
