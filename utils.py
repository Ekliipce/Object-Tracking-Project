import cv2
import re

def draw_bounding_box(frame, id, bb_left, bb_top, bb_width, bb_height, color, thickness):
    x1 = int(bb_left)
    y1 = int(bb_top)
    x2 = int(bb_left + bb_width)
    y2 = int(bb_top + bb_height)
    text = f"ID : {id}"
    image = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)

def get_detlines(file):
    # Each line represents one object instance and contains 10 values
    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>]
    with open(file) as f:
        det_lines = f.readlines()

    det_lines = [re.split(",|\n", lines, 9) for lines in det_lines]
    det_lines = [[float(x) for x in line] for line in det_lines] 
    values_name = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
    det_lines = [dict(zip(values_name, line)) for line in det_lines]
    
    return det_lines