# Save this as generate_priors.py and run: python generate_priors.py
input_file = "contigs.txt"
output_file = "contig_ploidy_priors.tsv"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    outfile.write("contig\tploidy\tprior\n")
    for line in infile:
        contig = line.strip()
        if contig in ["chrX"]:
            outfile.write(f"{contig}\t2\t0.5\n")
            outfile.write(f"{contig}\t1\t0.5\n")
        elif contig in ["chrY"]:
            outfile.write(f"{contig}\t1\t0.5\n")
            outfile.write(f"{contig}\t0\t0.5\n")
        elif contig in ["chrM", "MT"]:
            outfile.write(f"{contig}\t1\t1.0\n")
        else:
            outfile.write(f"{contig}\t2\t1.0\n")
