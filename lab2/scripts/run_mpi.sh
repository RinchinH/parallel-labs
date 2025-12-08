#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EXE="$HERE/../mpi/jacobi_mpi"

# Загружаем MPICH в текущей оболочке — env унаследует srun
module purge >/dev/null 2>&1 || true
module load mpi/mpich-x86_64 >/dev/null 2>&1 || true

N="${1:-1000000}"
EPS="${2:-1e-8}"
MAXIT="${3:-10000}"
MODE="fast"

out_dir="$HERE/../results"
mkdir -p "$out_dir"
csv="$out_dir/timings_mpi.csv"
echo "ranks,N,mode,eps,maxit,time_sec,iters,residual,rel_err" > "$csv"

run_one() {
  local ranks="$1" nodes="$2" ntpn="$3"
  local tmp="$(mktemp)"
  # Сливаем stdout+stderr в файл, чтобы ничего лишнего не печаталось в терминал
  srun --export=ALL -p compclass -N "$nodes" -n "$ranks" --ntasks-per-node="$ntpn" \
      "$EXE" -n "$N" -eps "$EPS" -k "$MAXIT" -q >"$tmp" 2>&1 || true
  # Берём первую корректную строку формата "time,iters,residual,rel_err"
  local core="$(grep -E '^[0-9]+\.[0-9]+,[0-9]+,' "$tmp" | head -n1)"
  if [ -z "$core" ]; then
    echo "RUN ranks=$ranks: нет корректной строки в выводе, см. $tmp" >&2
    return 1
  fi
  echo "$ranks,$N,$MODE,$EPS,$MAXIT,$core" | tee -a "$csv"
  rm -f "$tmp"
}

# 1,2,4,8 на одном узле
run_one 1 1 1
run_one 2 1 2
run_one 4 1 4
run_one 8 1 8
# 16 = 8+8 на двух узлах
run_one 16 2 8

echo "Done -> $csv"
