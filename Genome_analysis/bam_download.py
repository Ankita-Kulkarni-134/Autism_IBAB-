import subprocess
import pandas as pd
import os

# --- Configuration ---
csv_file = "/home/group_shyam01/Desktop/Autism_IBAB/genome_analysis/bam_cnv/sample_bam.csv"  # Your CSV file with SRR IDs
sra_path = os.path.expanduser("/home/group_shyam01/Desktop/Autism_IBAB/genome_analysis/bam_cnv")  # Path where prefetch stores files

# --- Read CSV File ---
df = pd.read_csv(csv_file)
srr_ids = df.iloc[:, 0].tolist()  # assuming SRR IDs are in the first column

# --- Loop over SRR IDs ---
for srr in srr_ids:
    print(f"Processing {srr}...")

    # Step 1: Download SRA file using prefetch
    try:
        subprocess.run(["prefetch", srr], check=True)
        print(f"{srr} download complete.")
    except subprocess.CalledProcessError:
        print(f"Failed to prefetch {srr}")
        continue

    # Step 2: Convert to BAM using sam-dump and samtools
    sra_file_path = os.path.join(sra_path, f"{srr}.sra")
    bam_output_file = f"{srr}.bam"

    try:
        with open(bam_output_file, 'wb') as bam_out:
            samdump = subprocess.Popen(
                ["sam-dump", "--aligned-region", sra_file_path],
                stdout=subprocess.PIPE
            )
            samtools = subprocess.Popen(
                ["samtools", "view", "-bS", "-"],
                stdin=samdump.stdout,
                stdout=bam_out
            )
            samdump.stdout.close()  # Allow samdump to receive a SIGPIPE if samtools exits.
            samtools.communicate()
        print(f"{srr}.bam created.")
    except Exception as e:
        print(f"Error processing {srr}: {e}")

