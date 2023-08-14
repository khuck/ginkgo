#!/bin/bash -l

declare -a SOLVERS=("bicgstab" "cg")
declare -a BATCH_SIZES=("8192" "16384" "32768" "65536" "131072")
declare -a MAT_SIZES=("16" "32" "64" "128" "256" "512" "1024" "2048")
declare -a NUM_TILES_LIST=("2" "1")

EXEC="dpcpp"
VER="pvc"

BIN_PREFIX_PATH="../../build/examples/batched-solver"
OUTPUT_PATH="performance"
DIR="${OUTPUT_PATH}/${VER}"
mkdir -p $DIR

#Warm up if submitting as a job
tmp=$(${BIN_PREFIX_PATH}/batched-solver $EXEC 131072 128 cg time no_resid)


for NUM_TILES in "${NUM_TILES_LIST[@]}"
do
  if [ $NUM_TILES -eq 1 ];  then
    export ZE_AFFINITY_MASK=0.0
  else
    unset ZE_AFFINITY_MASK
  fi

  for SOLVER in "${SOLVERS[@]}"
  do
    FNAME="$DIR/${EXEC}_${NUM_TILES}t_batch_${SOLVER}.txt"
    echo -e "Writing runtime data into $FNAME ..."
    echo -e "b\m ${MAT_SIZES[*]}" | tr " " "\t" >> $FNAME
    for b in "${BATCH_SIZES[@]}"
    do
      write_buffer=("${b}")
      for m in "${MAT_SIZES[@]}"
      do
        echo "Running ./batched-solver $EXEC $b $m $SOLVER time no_resid"
        runtime=$(${BIN_PREFIX_PATH}/batched-solver $EXEC $b $m $SOLVER time no_resid)
        write_buffer+=("$runtime")
      done
      echo -e "${write_buffer[*]}" | tr " " "\t" >> $FNAME
    done
  done
done
