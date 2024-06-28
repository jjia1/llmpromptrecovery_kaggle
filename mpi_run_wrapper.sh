#!/bin/bash
module load openmpi/gcc/64/4.1.5
module load cuda12.2/toolkit/12.2.2
export PYTHONUNBUFFERED=1
mpirun -np 1 --output-filename ${1%.*}/output --prefix /cm/shared/apps/openmpi4/gcc/4.1.5 --host sabercore-a100-002 /home/johnathanj/.conda/envs/llama2/bin/python  $1

