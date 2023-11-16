#. ./generate_images.sh --debug --n_points 10 --exclude_objects "3 5"  --changed_mechs "shape35"
#. ./generate_images.sh --debug --n_points 10 --exclude_objects "0 1 2 4 6" --changed_mechs "shape01246"
# Above with a for loop for exclude_objects:
for exclude_objects in "3 5" "0 1 2 4 6"; do
   (
   nospace_exclude_objects=$(echo "$exclude_objects" | tr -d ' ')
   for thing_to_skip in "latent_creation"; do
    . ./generate_images.sh --n_points 100000 --changed_mechs "shape_no_${nospace_exclude_objects}" --exclude_objects "$exclude_objects" --skip_${thing_to_skip}
    done
    ) &
done