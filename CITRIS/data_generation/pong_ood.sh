#!/bin/bash
DATASET_SIZE=10000
OUT_DIR=ood_pong
NUM_PROCESSES=8
OOD=()
maybe_debug=
maybe_no_intvs=
maybe_no_noise=
maybe_paper_settings=
# Parse arguments
while [[ $# -gt 0 ]]; do
      key="$1"
      case $key in
          -o|--output_folder)
              OUT_DIR="$2"
              shift # past argument
              shift # past value
              ;;
          -d|--dataset_size)
              DATASET_SIZE="$2"
              shift # past argument
              shift # past value
              ;;
          -p|--num_processes)
              NUM_PROCESSES="$2"
              shift # past argument
              shift # past value
              ;;
          --ood)
              OOD+=("$2")
              shift # past argument
              shift # past value
              ;;
          --no_intvs)
              maybe_no_intvs="--no_intvs"
              shift # past argument
              ;;
          --no_noise)
              maybe_no_noise="--no_noise"
              shift # past argument
              ;;
          --paper_settings)
            maybe_paper_settings="--paper_settings"
            shift # past argument
            ;;
          --debug)
          maybe_debug=echo
          shift # past argument
          ;;
          -h|--help)
              echo "Usage: pong_ood.sh [-o|--output_folder] [-d|--dataset_size] [-p|--num_processes] [--ood] [--no_intvs] [--no_noise] [--debug]"
              exit 0
              ;;
          *)    # unknown option
              shift # past argument
              ;;
      esac
done

# if no OOD is specified, use all of them
if [ ${#OOD[@]} -eq 0 ]; then
    OOD=("ball_x" "ball_y" "ball_vel_magn" "ball_vel_dir" "paddle_left_y" "paddle_right_y" "score_left" "score_right")
fi

for ood in "${OOD[@]}"; do
    echo "Generating OOD data for $ood"
    $maybe_debug python data_generation_interventional_pong.py --output_folder $OUT_DIR --dataset_size $DATASET_SIZE --num_processes $NUM_PROCESSES --ood $ood $maybe_no_intvs $maybe_no_noise $maybe_paper_settings &
done
wait