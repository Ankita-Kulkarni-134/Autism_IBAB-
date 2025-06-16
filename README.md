# Autism_IBAB

This repository aims to build machine learning models to predict Autism using copy number variation (CNV) calls from genomic samples. It covers the full workflow from genome analysis to predictive modeling, integrating bioinformatics and machine learning techniques to uncover genomic patterns associated with Autism.

Autism_IBAB/
├── genome_analysis/

│ ├── bam_download.py # Script for downloading mapped BAM files

│ ├── contig_ploidy_priors.py # Script to create contig ploidy priors

│ ├── fastq.py # Downloading FASTQ files using SRA Toolkit

│ ├── fastq_analysis.py # Analyzing FASTQ sequences

│ ├── merged_bed_files.sh # Bash script to merge BED files

│ ├── parallel_fastqc.py # Parallel processing for FastQC

│ ├── remove_duplicates.py # Script to remove duplicate samples

│ └── README.md # Documentation for genome analysis


├── machine_learning/

│ ├── Linear_regression_Theory_pipeline.pdf # Theoretical pipeline for linear regression

│ ├── Linear_regression_script.py # Implementation of linear regression

│ ├── Logistic_reg.py # Logistic regression implementation

│ ├── cross_validation/ # Manual cross-validation implementation

│ └── README.md # Documentation for machine learning


├── python_tasks/

│ ├── tasks/ # Python exercises and scripts for practice

│ └── README.md # Documentation for Python utilities


---

## 🎯 Project Objective

To leverage CNV data derived from individuals with Autism to build robust machine learning models for predictive diagnosis. This project integrates bioinformatics workflows (e.g., FASTQ and BAM processing, variant calling) with classical machine learning techniques (e.g., linear/logistic regression) for pattern recognition and classification.

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ankita-Kulkarni-134/Autism_IBAB-.git
   cd Autism_IBAB-
