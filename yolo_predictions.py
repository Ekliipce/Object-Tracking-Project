import cv2
import torch
import os
import argparse

parser = argparse.ArgumentParser(description="Yolo Detection")
parser.add_argument('--video_input', type=str, required=True, help='Path to input video frames')
parser.add_argument('--output_file', type=str, required=True, help='Path to output solution file')

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
parser = parser.parse_args()
list_dir = os.listdir(parser.video_input)
list_dir.sort()



with open(parser.output_file , "w+") as f:
    for i, file_name in enumerate(list_dir):
        file_path = os.path.join(parser.video_input, file_name)
        img = cv2.imread(file_path)
        results = model(img)

        pietons = results.xyxy[0][results.xyxy[0][:, -1] == 0]

        for pieton in pietons:
            x1, y1, x2, y2 = map(int, pieton[:4])
            w, h = x2 - x1, y2 - y1
            conf = pieton[4]
            
            if conf > 0.5:
                f.write(f"{i+1},-1,{x1},{y1},{w},{h},{conf},-1,-1,-1\n")
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         # Afficher l'image
#         cv2.imshow('Image', img)
#         cv2.waitKey(0)  # Attendre une touche pour passer à l'image suivante

# cv2.destroyAllWindows()
