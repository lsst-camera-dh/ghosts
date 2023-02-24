#!/bin/bash

export CONDA_DIR="/pbs/home/b/bregeon/lsst/conda"
export PATH="${CONDA_DIR}/bin:${PATH}"

if [ -n "${LD_LIBRARY_PATH}" ]; then
    export LD_LIBRARY_PATH="${CONDA_DIR}/lib:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${CONDA_DIR}/lib"
fi

source ${CONDA_DIR}/etc/profile.d/conda.sh

conda activate ghosts 

exec python -m ipykernel_launcher "$@"

