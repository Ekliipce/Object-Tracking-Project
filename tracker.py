import numpy as np
import cv2
import re

from Matching import Matcher
from utils import draw_bounding_box, get_detlines

# det_file_name = "ADL-Rundle-6/det/det.txt"
det_file_name = "ADL-Rundle-6/gt/gt.txt"
det_lines = get_detlines(det_file_name)

matcher = Matcher()

i = 1
with open("ADL-Rundle-6/det/sol.txt", "w+") as f:
    while True:
        frame_name = f"ADL-Rundle-6/img1/{i:06d}.jpg"
        frame = cv2.imread(frame_name)

        current_frames = [line for line in det_lines if line["frame"] == i]
        for fr in current_frames:
            fr["id"] = -1
        matcher.set_currentframes(current_frames)
        lines, tracks = matcher.find_matching_id(init=True if i==1 else False)

        for line, track  in zip(lines, tracks):
            draw_bounding_box(frame, line["id"], line["bb_left"], line["bb_top"],
                              line["bb_width"], line["bb_height"], (0, 0, 255), 2)
            f.write(f"{line['frame']},{line['id']},{line['bb_left']},{line['bb_top']},{line['bb_width']},{line['bb_height']}\n")

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or i > len(det_lines):
            break
        # while True:
        #     if cv2.waitKey(1) & 0xFF == ord('p'):
        #         break
        i += 1