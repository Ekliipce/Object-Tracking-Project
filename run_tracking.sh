#!/bin/bash

# Liste des répertoires
declare -a dirs=("ADL-Rundle-8" "ETH-Bahnhof" "ETH-Pedcross2" "ETH-Sunnyday" "KITTI-13" "KITTI-17" "PETS09-S2L1" "TUD-Campus" "TUD-Stadtmitte" "Venice-2")
# declare -a dirs=("KITTI-13")

# Chemin de base pour les répertoires
base_path="evaluation/MOT15/train"

for dir in "${dirs[@]}"; do
    echo "Processing directory: $dir"

    python3 yolo_predictions.py --video_input "$base_path/$dir/img1" --output_file "$base_path/$dir/det/yolo.txt"
    # python3 tracker.py --det_file "$base_path/$dir/det/det.txt" --video_input "$base_path/$dir/img1/" --output_video "$base_path/$dir/output.avi" --output_file "$base_path/$dir/$dir.txt"
    python3 tracker.py --det_file "$base_path/$dir/det/yolo.txt" --video_input "$base_path/$dir/img1/" --output_video "$base_path/$dir/output.avi" --output_file "$base_path/$dir/$dir.txt"

    cp "$base_path/$dir/$dir.txt" "$base_path/../../TrackEval-master/data/trackers/mot_challenge/MOT15-train/MyTracker/data/"

done