import cv2
from utils import get_detlines, draw_bounding_box
from Matching import Matcher
from torchvision.io import read_image
import torch


def process_frame(frame, matcher, current_frames, frame_index, video_writer, output_file):
    matcher.set_currentframes(current_frames, torch.from_numpy(frame).permute(2, 0, 1))
    lines, _ = matcher.find_matching_id(init=frame_index == 1)

    for line in lines:
        draw_bounding_box(frame, line["id"], line["bb_left"], line["bb_top"],
                          line["bb_width"], line["bb_height"], (0, 0, 255), 2)
        output_file.write(f"{frame_index},{line['id']},{line['bb_left']},{line['bb_top']},{line['bb_width']},{line['bb_height']}\n")

    video_writer.write(frame)

def main():
    det_file_name = "ADL-Rundle-6/det/det.txt"
    det_lines = get_detlines(det_file_name)

    matcher = Matcher()
    cap = cv2.VideoCapture("ADL-Rundle-6/img1/%06d.jpg")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30  # Adapt according to your video
    success, frame = cap.read()
    if not success:
        print("Failed to read video")
        return
    print(frame.shape)

    frame_size = (frame.shape[1], frame.shape[0])
    out = cv2.VideoWriter('output.avi', fourcc, fps, frame_size)

    frame_index = 1
    with open("ADL-Rundle-6/det/sol.txt", "w+") as f:
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