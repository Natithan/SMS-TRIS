# NOTE: this is deprecated in favor of generate_ood_data.py

#P2_ROOT="/mnt/g/My Drive/1 PhD/1 p2/p2" # Assume this is specified in bashrc
#echo "P2_ROOT=$P2_ROOT"
CITRIS_ROOT="$P2_ROOT/CITRIS"
BLENDER_OUT="${CITRIS_ROOT}/data_generation/temporal_causal3dident"
echo "Changing to Blender output directory: ${BLENDER_OUT}"
cd "${BLENDER_OUT}" || kill -INT $$

N_POINTS=100
#CHANGED_MECHS=("x_pos" "y_pos" "z_pos" "alpha" "beta" "rot_spotlight" "hue_object" "hue_spotlight" "hue_background" )
CHANGED_MECHS="x_pos y_pos z_pos alpha beta rot_spotlight hue_object hue_spotlight hue_background"
DO_AT_ONCE=0
SKIP_LATENT_CREATION=0
SKIP_IMAGE_GENERATION=0
intv_arg=
intv_suffix=
maybe_debug=
exclude_objects=
batch_size=2500
while [[ $# -gt 0 ]]
do
  key="$1"
#  echo "key"
#  echo $key

  case $key in
      --n_points)
      N_POINTS=$2
      shift # past argument
      shift # past value
      ;;
      --changed_mechs)
      CHANGED_MECHS=$2
      shift # past argument
      shift # past value
      ;;
      --debug)
      maybe_debug=echo
      shift # past argument
      ;;
      --do_at_once)
      DO_AT_ONCE=1
      shift # past argument
      ;;
      --skip_latent_creation)
      SKIP_LATENT_CREATION=1
      shift # past argument
      ;;
      --skip_image_generation)
      SKIP_IMAGE_GENERATION=1
      shift # past argument
      ;;
#      --intv_prob)
#      intv_prob="--intv_prob $2"
#      shift # past argument
#      shift # past value
#      ;;
      --no_intv)
      intv_arg="--no_intv"
      intv_suffix="_no_intv"
      shift # past argument
      ;;
    # in data_generation_causal3dident.py: parser.add_argument('--exclude_objects', type=int, nargs='+', default=None, help='List of objects to exclude for generating the dataset, e.g. to test the generalization to unseen shapes.')
      --exclude_objects)
      excluded_objects="$2"
      shift # past argument
      shift # past value
      ;;
    # per-process batch size
      --batch_size)
      batch_size="$2"
      shift # past argument
      shift # past value
      ;;
      *) echo "Unknown parameter passed: $key"; kill -INT $$ ;;
  esac
done

function join_by {
#  From https://stackoverflow.com/a/17841619
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}

generate () {
  local_changed_mechs="$1"
  mechs_string=$(join_by __ $local_changed_mechs)
  DETAILED_OUTPUT_FOLDER="${BLENDER_OUT}/${N_POINTS}_points${intv_suffix}/${mechs_string}"
  echo "Output folder: ${DETAILED_OUTPUT_FOLDER}"
  for split in train; do
#  for split in val; do
#  for split in train val test; do
    (echo "Split: ${split}"
    # If split is test or val, cap N_POINTS at n_points_capped=min(N_POINTS, 10000)
    if [[ "$split" == "train" ]]; then
      n_points_capped=$N_POINTS
    else
      n_points_capped=10000
      if [[ $N_POINTS -lt $n_points_capped ]]; then
        n_points_capped=$N_POINTS
      fi
    fi
    echo "n_points_capped: $n_points_capped"
    # if local_changed_mechs starts with "shape", then OOD is empty, otherwise OOD is "--ood ${local_changed_mechs}"
    if [[ "$local_changed_mechs" == "shape"* ]]; then
      ood_arg=
    else
      ood_arg="${local_changed_mechs}"
    fi


    if [[ $SKIP_LATENT_CREATION -eq 0 ]]; then
      # shellcheck disable=SC2086
      $maybe_debug python data_generation_causal3dident.py --n_points "$n_points_capped" --output_folder "${DETAILED_OUTPUT_FOLDER}" --coarse_vars  --num_shapes 7 --split $split $intv_arg --exclude_objects $excluded_objects --ood "$ood_arg"
    else
      echo "Skipping latent creation"
    fi

    if [[ $SKIP_IMAGE_GENERATION -eq 0 ]]; then
#      $maybe_debug "$P2_ROOT"/blender-2.93.8-linux-x64/blender --background --python "$BLENDER_OUT"/generate_causal3dident_images.py -- --output-folder "${DETAILED_OUTPUT_FOLDER}" --n-batches 1 --batch-index 0 --split $split
      # Image generation has a memory leak: 10k images uses about 40GB of RAM. Hence, split into batches of e.g. 5000, so 20GB RAM per parallel generation, so for 9 generations 180GB.
      # If batch_size > n_points_capped, then n_batches=1
      # Else, n_batches = n_points_capped / batch_size
      if [[ $n_points_capped -lt $batch_size ]]; then
        n_batches=1
      else
        n_batches=$((n_points_capped / batch_size))
      fi
      echo "n_batches: $n_batches"
      for batch_index in $(seq 0 $((n_batches - 1))); do
        $maybe_debug "$P2_ROOT"/blender-2.93.8-linux-x64/blender --background --python "$BLENDER_OUT"/generate_causal3dident_images.py -- --output-folder "${DETAILED_OUTPUT_FOLDER}" --n-batches $n_batches --batch-index $batch_index --split $split &
      done
      wait
      $maybe_debug python npzify.py --dir "${DETAILED_OUTPUT_FOLDER}" --split $split
    else
      echo "Skipping image generation and npzification"
    fi)
done
}

# Print separating line with "="
echo "=================================================================================================================================================================================================================="
if [[ $DO_AT_ONCE -eq 1 ]]; then
  echo "Changing multiple mechanisms in one dataset"
  generate "$CHANGED_MECHS"
else
  echo "Changing one mechanism in multiple datasets"
  for changed_mech in $CHANGED_MECHS; do
    (echo "Changed mechanism: ${changed_mech}"
    generate "$changed_mech") &
  done
fi

# Example usage: ./generate_images.sh --n_points 2 --debug --changed_mechs "x_pos y_pos z_pos alpha beta rot_spotlight hue_object hue_spotlight hue_background"