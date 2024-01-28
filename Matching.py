import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from KalmanFilter import KalmanFilter
from EmbeddingSimilarity import EmbeddingSimilarity
from memory_profiler import profile

class Matcher():
    def __init__(self, w_iou=0.5, w_similarity=0.5, longevity=20):
        self.current_frames_info = []
        self.current_frames = None
        self.track = []
        self.kalman_filters = {}
        self.associations = {}
        self.embedding_similarity = EmbeddingSimilarity()
        self.w_iou = w_iou
        self.w_similarity = w_similarity
        self.longevity = longevity
        self.nb_track = 0
        self.set_id = set()

    def set_currentframes(self, current_frames_info, current_frames):
        self.current_frames_info = current_frames_info
        self.current_frames = current_frames

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Compute the area of both bounding boxes
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou
    
    def create_kalman_filter(self):
        dt = 0.1
        u_x, u_y = 1, 1
        std_acc = 1 
        x_std_meas, y_std_meas = 0.1, 0.1 
        return KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    
    def convert_bb_to_centroid(self, bb_left, bb_top, bb_width, bb_height):
        return np.array([bb_left + bb_width/2, bb_top + bb_height/2])
    
    def convert_centroid_to_bb(self, centroid, width, height):
        return np.array([centroid[0] - width/2, centroid[1] - height/2, centroid[0] + width/2, centroid[1] + height/2])

    def hungarian_similarity_matrix(self):
        detections = self.current_frames_info
        tracks = self.track

        num_detections = len(detections)
        num_tracks = len(tracks)

        high_cost = 1e5

        # max_size = max(num_detections, num_tracks)
        cost_matrix = np.full((num_detections, num_tracks), high_cost)

        embedding_batch_detection = self.embedding_similarity.compute_batch_embedding(self.current_frames, detections)
        embedding_batch_track = self.embedding_similarity.compute_batch_embedding(self.current_frames, tracks)

        for d, detection in enumerate(detections):
            d_x1, d_y1 = int(detection["bb_left"]), int(detection["bb_top"])
            d_w1, d_h1 = int(detection["bb_width"]), int(detection["bb_height"])
            blockA = [d_x1, d_y1, d_x1 + d_w1, d_y1 + d_h1]
            emb1 = embedding_batch_detection[d].unsqueeze(0)

            for t, track in enumerate(tracks):
                t_x2, t_y2 = int(track["bb_left"]), int(track["bb_top"])
                t_w2, t_h2 = int(track["bb_width"]), int(track["bb_height"])
                blockB = self.kalman_filters[track["id"]].predict()
                blockB = self.convert_centroid_to_bb(blockB, t_w2, t_h2)

                # Compute visual similarity between the box
                emb2 = embedding_batch_track[t].unsqueeze(0)
                
                similarity = self.embedding_similarity.compute_similarity(embedding1=emb1, embedding2=emb2)

                iou = self.compute_iou(blockA, blockB)
                cost_matrix[d, t] = 1 - (self.w_iou * iou + self.w_similarity * similarity)
        del embedding_batch_detection
        del embedding_batch_track
        
        # Application de l'algorithme hongrois
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        return cost_matrix, row_ind, col_ind
    
    def associate_detections_to_tracks(self):
        cost_matrix, row_ind, col_ind = self.hungarian_similarity_matrix()
        keep_track = []
        self.associations = {}
        
        for d, t in zip(row_ind, col_ind):
            # Mise à jour des tracks existants
            self.associations[d] = t
            self.current_frames_info[d]["id"] = self.track[t]["id"]
            keep_track.append(self.track[t])

            # Update Kalman filter
            center = self.convert_bb_to_centroid(self.current_frames_info[d]["bb_left"],
                                            self.current_frames_info[d]["bb_top"], 
                                            self.current_frames_info[d]["bb_width"],
                                            self.current_frames_info[d]["bb_height"])
            if self.track[t]["id"] in self.kalman_filters:
                self.kalman_filters[self.track[t]["id"]].update(center)
            else:
                self.kalman_filters[self.track[t]["id"]] = self.create_kalman_filter()

        #Remove tracks with no matching detections
        for track in self.track:
            if track["id"] not in [t["id"] for t in keep_track]:
                track["longevity"] -= 1
                if track["longevity"] <= 0:
                    del self.kalman_filters[track["id"]]
                    self.set_id.remove(track["id"])
                else:
                    keep_track.append(track)
                
        #Create new tracks for unmatched detections
        for frames in self.current_frames_info:
            if frames["id"] == -1:
                frame_id = max(self.set_id, default=0) + 1
                frames["id"] = frame_id
                frames["longevity"] = self.longevity
                self.kalman_filters[frames["id"]] = self.create_kalman_filter()
                keep_track.append(frames)
                self.set_id.add(frame_id)
                self.nb_track += 1


        self.track = keep_track

    def find_matching_id(self, init=False):
        if init:
            for num_line, line in enumerate(self.current_frames_info):
                x, y = int(line["bb_left"]), int(line["bb_top"])
                w, h = int(line["bb_width"]), int(line["bb_height"])
                line["id"] = int(num_line)
                line["longevity"] = self.longevity
                self.kalman_filters[num_line] = self.create_kalman_filter()
                self.set_id.add(int(num_line))

            self.track = self.current_frames_info
            self.nb_track = len(self.track)
        else:
            self.associate_detections_to_tracks()
            

        return self.current_frames_info, self.track

    
