import numpy as np
import argparse
import os
import struct

def parse_shr_file(file_path):
    with open(file_path, 'rb') as f:
        header_size = 16384
        header = f.read(header_size)
        trace_data = np.fromfile(f, dtype=np.float32)
        
    return header, trace_data

def save_to_txt(header, data, output_path):
    """
    Save the parsed header and numpy array data to a text file
    
    Args:
        header: Binary header data from SHR file
        data: Numpy array containing the parsed data
        output_path: Path to save the text file
    """
    with open(output_path, 'w') as f:
        # Write header information
        f.write("# SHR FILE HEADER INFORMATION\n")
        f.write("# --------------------------\n")
        
        # Try to extract some meaningful information from the header
        f.write("# Header size: 16384 bytes\n")
        f.write("# Header hex dump (first 100 bytes):\n")
        for i in range(0, min(100, len(header)), 16):
            hex_vals = ' '.join(f'{b:02x}' for b in header[i:i+16])
            ascii_vals = ''.join(chr(b) if 32 <= b < 127 else '.' for b in header[i:i+16])
            f.write(f"# {i:04x}: {hex_vals} | {ascii_vals}\n")
            
        f.write("\n# DATA SECTION\n")
        f.write(f"# SHR file data - {len(data)} samples\n")
        f.write("# Index, Value\n")
        
        # Write each data point with its index
        for i, value in enumerate(data):
            f.write(f"{i}, {value}\n")
    
    print(f"Data successfully written to {output_path}")

def analyze_header(header):
    """
    Analyze the SHR file header and print important information
    
    Args:
        header: Binary header data from SHR file
    """
    print("Header Information:")
    print("-----------------")
    print(f"Header Size: 16384 bytes")
    
    # Try to find strings in the header
    printable_chars = []
    for i in range(len(header)):
        if 32 <= header[i] < 127:  # ASCII printable characters
            printable_chars.append(chr(header[i]))
        else:
            if printable_chars and len(printable_chars) > 3:  # Only print strings of length > 3
                print(f"String at offset {i-len(printable_chars)}: {''.join(printable_chars)}")
            printable_chars = []
    
    # Print first few bytes as hex
    print("\nHeader first 64 bytes (hex):")
    for i in range(0, min(64, len(header)), 16):
        hex_vals = ' '.join(f'{b:02x}' for b in header[i:i+16])
        ascii_vals = ''.join(chr(b) if 32 <= b < 127 else '.' for b in header[i:i+16])
        print(f"{i:04x}: {hex_vals} | {ascii_vals}")

def main():
    # Set up argument parser for command line usage
    parser = argparse.ArgumentParser(description='Parse SHR file and convert to text')
    parser.add_argument('--input_file', help='Path to the SHR file to parse')
    parser.add_argument('-o', '--output', help='Path to output text file (default: input_filename.txt)')
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return
    
    # Set default output filename if not provided
    if args.output is None:
        base_name = os.path.splitext(args.input_file)[0]
        args.output = f"{base_name}.txt"
    
    # Parse the SHR file
    print(f"Parsing SHR file: {args.input_file}")
    header, trace_data = parse_shr_file(args.input_file)
    
    # Analyze and print header information
    analyze_header(header)
    
    # Save to text file
    print(f"Saving data to: {args.output}")
    save_to_txt(header, trace_data, args.output)
    
    print(f"Processed {len(trace_data)} samples")

if __name__ == "__main__":
    main()