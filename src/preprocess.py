# src/preprocess.py

import csv
import argparse
from collections import defaultdict

def parse_tsv(file_path):
    """Read a TSV file and return a list of dictionaries for each row."""
    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)
    return rows

def group_by_program(rows):
    """
    (Optional) Group rows by (probid, subid) to reconstruct full programs.
    Each key maps to a list of rows.
    """
    grouped = defaultdict(list)
    for r in rows:
        key = (r['probid'], r['subid'])
        grouped[key].append(r)
    return grouped

def preprocess_line(r):
    """For a single row, choose text if available; otherwise, fallback to code."""
    pseudo = r['text'].strip() if r['text'].strip() else r['code'].strip()
    code_line = r['code'].strip()
    return pseudo, code_line

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_tsv', type=str, required=True,
                        help='Path to the TSV file (e.g., spoc-train-train.tsv)')
    parser.add_argument('--output_txt', type=str, required=True,
                        help='Output file to store paired pseudocode and code')
    args = parser.parse_args()

    rows = parse_tsv(args.input_tsv)
    
    # Option 1: Line-by-line pairing (simplest approach)
    with open(args.output_txt, 'w', encoding='utf-8') as out_f:
        for r in rows:
            pseudo, code_line = preprocess_line(r)
            out_f.write(f"{pseudo}\t{code_line}\n")
    
    # Option 2: Full-program grouping (uncomment if needed)
    # grouped = group_by_program(rows)
    # with open(args.output_txt, 'w', encoding='utf-8') as out_f:
    #     for key, program_rows in grouped.items():
    #         # Sort rows by 'line' to ensure proper order
    #         program_rows = sorted(program_rows, key=lambda x: int(x['line']))
    #         pseudo_program = "\n".join([preprocess_line(r)[0] for r in program_rows])
    #         code_program = "\n".join([r['code'].strip() for r in program_rows])
    #         out_f.write(f"{pseudo_program}\t{code_program}\n")

if __name__ == "__main__":
    main()
