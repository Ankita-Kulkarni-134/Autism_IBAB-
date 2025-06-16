#!/usr/bin/env bash
set -euo pipefail

# Merge all tabbed BED files
cat tabbed_*.bed \
  | sort -k1,1 -k2,2n \
  | awk '$1 ~ /^chr([1-9][0-9]?|X|Y)$/' \
  | bedtools merge -i - \
  > merged_exome_regions.bed

echo "Merged BED file saved to: merged_exome_regions.bed"
