import cv2
import torch
import os

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

list_dir = os.listdir("ADL-Rundle-6/img1")
list_dir.sort()

with open("ADL-Rundle-6/det/yolo.txt", "w+") as f:
    for i, file_name in enumerate(list_dir):
        file_path = os.path.join("ADL-Rundle-6/img1", file_name)
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
