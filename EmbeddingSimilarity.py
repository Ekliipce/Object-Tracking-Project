from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision import transforms
from torchvision.io import read_image

import torch.nn as nn
import PIL
import cv2
import torch


class EmbeddingSimilarity():
    def __init__(self):
        weights = EfficientNet_B1_Weights.IMAGENET1K_V2
        efficientnet = efficientnet_b1(weights=weights)
        self.model = nn.Sequential(*list(efficientnet.children())[:-1])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def compute_batch_embedding(self, img, boxes):
        self.model.eval()

        batch = []
        for i, elm in enumerate(boxes):
            x, y = max(int(elm["bb_left"]), 0), max(int(elm["bb_top"]), 0)
            w, h = min(int(elm["bb_width"]), img.shape[2] - x), min(int(elm["bb_height"]), img.shape[1] - y)
            img_cropped = img[:, y:y+h, x:x+w]
            img_cropped = transforms.ToPILImage()(img_cropped)
            img_cropped = self.transform(img_cropped) 
            batch.append(img_cropped)
        
        batch = torch.stack(batch)
        embedding = self.model(batch)
        return embedding


    def compute_embedding(self, img, x=None, y=None, width=None, height=None):
        self.model.eval()

        if (x is not None and y is not None) and \
           (width is not None and height is not None):
            x, y = max(x, 0), max(y, 0)
            w, h = min(width, img.shape[2] - x), min(height, img.shape[1] - y)
            
            img = img[:, y:y+h, x:x+w]

        img = transforms.ToPILImage()(img)
        img = self.transform(img)
        
        batch = img.unsqueeze(0)
        embedding = self.model(batch)
        return embedding
    
    def compute_similarity(self, embedding1=None, embedding2=None, img1=None, img2=None):
        if (img1 is None and embedding1 is None) or (img2 is None and embedding2 is None):
            raise ValueError("You must provide two images or two embeddings.")

        if embedding1 is None:
            embedding1 = self.compute_embedding(img1)
        if embedding2 is None:
            embedding2 = self.compute_embedding(img2)
        
        cosine_similarity = nn.CosineSimilarity()
        return cosine_similarity(embedding1, embedding2).item()


# # Example usage:
# # Decomment the following lines to test the EmbeddingSimilarity class

# jpg1 = cv2.imread("ADL-Rundle-6/img1/000001.jpg")
# jpg1= torch.from_numpy(jpg1).permute(2, 0, 1)
# print(jpg1.shape)
# print(type(jpg1))
# # jpg1 = read_image("ADL-Rundle-6/img1/000001.jpg")
# img = jpg1[:, 385:385+339, 1703:1703+157]
# # img_pil = transforms.ToPILImage()(img)  # Conversion du tenseur en image PIL
# PIL.Image.fromarray(img.permute(1, 2, 0).numpy()).show()

# jpg2 = cv2.imread("ADL-Rundle-6/img1/000002.jpg")
# jpg2 = torch.from_numpy(jpg2).permute(2, 0, 1)
# print(type(jpg2))
# print(jpg2.shape)
# # 2,1,1699,383,159,341,1,-1,-1,-1
# img1 = jpg2[:, 383:383+341, 1699:1699+159 ]
# PIL.Image.fromarray(img1.permute(1, 2, 0).numpy()).show()


# # 2,3,1293,455,83,213,1,-1,-1,-1
# img2 = jpg2[:, 455:455+213, 1293:1293+83]
# PIL.Image.fromarray(img2.permute(1, 2, 0).numpy()).show()


# emb = EmbeddingSimilarity()
# embedding2 = emb.compute_embedding(jpg2, x=1699, y=383, width=159, height=341)
# print(emb.compute_similarity(img1=img, embedding2=embedding2))
# print(emb.compute_similarity(img1=img, img2=img2))
