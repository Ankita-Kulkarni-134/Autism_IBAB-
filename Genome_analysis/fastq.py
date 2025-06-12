import pandas as pd
import subprocess
import time
import os
import concurrent.futures

# Path to the CSV file
csv_path = "/home/group_shyam01/Desktop/Autism_IBAB/datasets/CONCLUDED/concluded_2/fastq/fastq_300above.csv"

# Output directory for downloads
output_dir = "/home/group_shyam01/Desktop/Autism_IBAB/genome_analysis/fastq_300"
os.makedirs(output_dir, exist_ok=True)

# Load the CSV
df = pd.read_csv(csv_path)

# Check for 'Run' column
if 'Run' not in df.columns:
    raise ValueError("The 'Run' column is not found in the CSV file.")

# Extract unique SRA IDs
sra_ids = df['Run'].dropna().unique()


# Function to run the fastq-dump command
def download_sra(sra_id, idx):
    command = [
        "fastq-dump",
        "--split-files",
        "--gzip",
        "--outdir", output_dir,
        sra_id
    ]
    print(f"Running command {idx}: {' '.join(command)}")
    subprocess.run(command)

    # Pause after every 3 downloads
    if idx % 3 == 0:
        print("Pausing for 5 seconds...\n")
        time.sleep(5)


# Use ThreadPoolExecutor to run downloads concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    # Submit the download tasks to the thread pool
    futures = [executor.submit(download_sra, sra_id, idx) for idx, sra_id in enumerate(sra_ids, 1)]

    # Wait for all tasks to complete
    concurrent.futures.wait(futures)

print("All downloads completed.")
