import cv2
import torch
import argparse
import os

from utils import get_detlines, draw_bounding_box
from Matching import Matcher
from torchvision.io import read_image


def parse_args():
    parser = argparse.ArgumentParser(description="Object Tracking with Matcher")
    parser.add_argument('--det_file', type=str, required=True, help='Path to detection file')
    parser.add_argument('--video_input', type=str, required=True, help='Path to input video frames')
    parser.add_argument('--output_video', type=str, default='output.avi', help='Path to output video')
    parser.add_argument('--output_file', type=str, default='sol.txt', help='Path to output solution file')
    parser.add_argument('--iou_weight', type=float, default=0.3, help='Weight for IOU in matcher')
    parser.add_argument('--similarity_weight', type=float, default=0.7, help='Weight for similarity in matcher')

    return parser.parse_args()


def process_frame(frame, matcher, current_frames, frame_index, video_writer, output_file):
    matcher.set_currentframes(current_frames, torch.from_numpy(frame).permute(2, 0, 1))
    lines, _ = matcher.find_matching_id(init=frame_index == 1)

    for line in lines:
        draw_bounding_box(frame, line["id"], line["bb_left"], line["bb_top"],
                          line["bb_width"], line["bb_height"], (0, 0, 255), 2)
        output_file.write(f"{frame_index},{line['id']},{line['bb_left']},{line['bb_top']},{line['bb_width']},{line['bb_height']},1,-1,-1,-1\n")

    video_writer.write(frame)

def main():
    args = parse_args()

    det_file_name = args.det_file
    det_lines = get_detlines(det_file_name)

    matcher = Matcher(w_iou=args.iou_weight, w_similarity=args.similarity_weight)
    cap = cv2.VideoCapture(os.path.join(args.video_input, "%06d.jpg"))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30  # Adapt according to your video
    success, frame = cap.read()
    if not success:
        print("Failed to read video")
        return
    print(frame.shape)

    frame_size = (frame.shape[1], frame.shape[0])

    out = cv2.VideoWriter(args.output_video, fourcc, fps, frame_size)

    frame_index = 1
    with open(args.output_file, "w+") as f:
        while success:
            current_frames = [line for line in det_lines if line["frame"] == frame_index]
            process_frame(frame, matcher, current_frames, frame_index, out, f)

            success, frame = cap.read()
            frame_index += 1
            print(f"Frame {frame_index} processed")

    cap.release()
    out.release()

if __name__ == "__main__":
    main()