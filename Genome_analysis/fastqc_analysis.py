import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
# Paths
fastqc_path = "/home/group_shyam01/Desktop/Autism_IBAB/genome_analysis/tools/FastQC/fastqc"
input_folder = "/home/group_shyam01/Desktop/Autism_IBAB/genome_analysis/1st_batch/fastq"
output_folder = "/home/group_shyam01/Desktop/Autism_IBAB/genome_analysis/1st_batch/fastq_before"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)
# Function to run FastQC on one file
def run_fastqc(file_name):
    input_file_path = os.path.join(input_folder, file_name)
    command = [
        fastqc_path,
        input_file_path,

        "-o", output_folder,
        "--threads", "2"
    ]
    print(f"Running FastQC on {file_name} ...")
    subprocess.run(command)
    print(f"✅ Completed: {file_name}")

# List all .fastq.gz files
fastq_files = [f for f in os.listdir(input_folder) if f.endswith(".fastq.gz")]

# Use ProcessPoolExecutor for parallel processing
with ProcessPoolExecutor(max_workers=5) as executor:
    executor.map(run_fastqc, fastq_files)

print(" All FastQC analyses completed using parallel processing.")
