import numpy as np
from scipy.optimize import linear_sum_assignment
from KalmanFilter import KalmanFilter

class Matcher():
    def __init__(self):
        self.current_frames_info = []
        self.track = []
        self.kalman_filters = {}
        self.associations = {}
        self.nb_tracks = 0

    def set_currentframes(self, current_frames_info):
        self.current_frames_info = current_frames_info

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

        for d, detection in enumerate(detections):
            for t, track in enumerate(tracks):
                blockA = [detection["bb_left"], detection["bb_top"], detection["bb_left"] + detection["bb_width"], detection["bb_top"] + detection["bb_height"]]
                blockB = self.kalman_filters[track["id"]].predict()
                blockB = self.convert_centroid_to_bb(blockB, track["bb_width"], track["bb_height"])

                iou = self.compute_iou(blockA, blockB)
                cost_matrix[d, t] = 1 - iou  # Conversion de l'IoU en coût

        # Application de l'algorithme hongrois
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        return cost_matrix, row_ind, col_ind
    
    def associate_detections_to_tracks(self):
        cost_matrix, row_ind, col_ind = self.hungarian_similarity_matrix()
        keep_track = []
        self.associations = {}
        
        #Keep only matching tracks and update info frame
        for d, t in zip(row_ind, col_ind):
            self.associations[d] = t
            self.current_frames_info[d]["id"] = self.track[t]["id"]
            keep_track.append(self.track[t])
            
            #Update Kalman filter
            center = self.convert_bb_to_centroid(self.current_frames_info[d]["bb_left"],
                                            self.current_frames_info[d]["bb_top"], 
                                            self.current_frames_info[d]["bb_width"],
                                            self.current_frames_info[d]["bb_height"])
            center = self.kalman_filters[self.track[t]["id"]].update(center)
        self.nb_tracks = len(keep_track)
                
        #Create new tracks for unmatched detections
        for frames in self.current_frames_info:
            if frames["id"] == -1:
                frames["id"] = max([current_frame["id"] for current_frame in self.current_frames_info]) + 1
                self.kalman_filters[frames["id"]] = self.create_kalman_filter()
                keep_track.append(frames)
                self.nb_tracks += 1


        self.track = keep_track

    def find_matching_id(self, init=False):
        if init:
            for num_line, line in enumerate(self.current_frames_info):
                line["id"] = int(num_line)
                self.kalman_filters[num_line] = self.create_kalman_filter()
            self.track = self.current_frames_info
            self.nb_tracks = len(self.current_frames_info)
        else:
            self.associate_detections_to_tracks()
            

        return self.current_frames_info, self.track

    
