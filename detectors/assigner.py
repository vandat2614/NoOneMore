from sklearn.cluster import KMeans
import numpy as np

import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_distance

class Assigner:

    def __init__(self):
        self.team1_color = None
        self.team2_color = None

        self.team_dict = {}

    def get_color(self, frame, bbox):
        
        img = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        img = img[ : int(img.shape[0] / 2), :]

        features = img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, random_state=42,
                             init="k-means++", n_init=1).fit(features)

        # get cluster center
        labels = kmeans.labels_
        cluster_img = labels.reshape(img.shape[0], img.shape[1])

        corner_labels = [cluster_img[0, 0], cluster_img[0, -1], cluster_img[-1, 0], cluster_img[-1, -1]]
        bg_label = max(set(corner_labels), key=corner_labels.count)
        player_label = 1 - bg_label

        player_color = kmeans.cluster_centers_[player_label]
        return player_color

    def fit(self, frame, frame_objects): # frame_objects is results[frame]["players"]
        
        player_color = []

        for player_id, info in frame_objects.items():
            bbox = info["bbox"]
            color = self.get_color(frame, bbox)
            player_color.append(color)

        # cluster colors to 2 groups

        self.kmeans = KMeans(n_clusters=2, random_state=42,
                        init="k-means++", n_init=1).fit(player_color)

        self.team1_color = self.kmeans.cluster_centers_[0]
        self.team2_color = self.kmeans.cluster_centers_[1]

    def predict(self, frame, bbox, player_id):

        if player_id in self.team_dict:
            return self.team_dict[player_id]
        
        color = self.get_color(frame, bbox)
        team_id = self.kmeans.predict(color.reshape(1, -1))[0]
        self.team_dict[player_id] = team_id + 1

        return team_id + 1

    def assign_ball_to_player(self, frame_result, ball_bbox):

        max_distance = 70
        ball_pos = get_center_of_bbox(ball_bbox)

        min_distance = None
        assign_id = None

        for player_id, info in frame_result.items():
            player_bbox = info["bbox"]

            left_distance = get_distance(
                (player_bbox[0], player_bbox[3]),
                ball_pos
            )
            right_distance = get_distance(
                (player_bbox[2], player_bbox[3]),
                ball_pos
            )

            distance = min(left_distance, right_distance)

            if distance < max_distance:
                if min_distance is None or distance < min_distance:
                    distance = min_distance
                    assign_id = player_id

        return player_id

