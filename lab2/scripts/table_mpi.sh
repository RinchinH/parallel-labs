#!/usr/bin/env bash
set -euo pipefail
csv="$(cd "$(dirname "$0")" && pwd)/../results/timings_mpi.csv"
awk -F, 'NF==9 && NR==2{t1=$6} NF==9 && NR>1{
  printf "| %-5s | %-8s | %-9.5f | %-6.2f | %-7.1f%% |\n",
         $1,$2,$6,t1/$6,(t1/$6)/$1*100
}' "$csv" | (echo "| Ranks | N       | Time (s)  | S     | Eff %  |";
             echo "|-------|---------|-----------|-------|--------|"; cat)
