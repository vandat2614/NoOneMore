from ultralytics import YOLO
import supervision as sv
import os
import pickle
import pandas as pd

class Detector:

    def __init__(self, weight_path):
        self.detector = YOLO(model=weight_path)
        self.tracker = sv.ByteTrack()

    def detect_object(self, frames, batch_size=32):
        processed_frames = [] 
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            batch_results = self.detector.predict(source=batch_frames, conf=0.1)
            processed_frames += batch_results
        return processed_frames
    
    def convert_to_sv_format(self, frame_result):
        return sv.Detections.from_ultralytics(ultralytics_results=frame_result)
    
    def assign_track_id(self, frame_in_sv_format):
        return self.tracker.update_with_detections(frame_in_sv_format)
        
    def process(self, frames, load=False, save_path=None):

        if load and save_path is not None and os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                results = pickle.load(f)

                return results

        processed_frames = self.detect_object(frames, batch_size=64)
        results = {}

        for frame_num, frame_with_objects in enumerate(processed_frames):

            cls_id_to_name = frame_with_objects.names
            cls_name_to_id = {v:k for k,v in cls_id_to_name.items()}

            sv_format = self.convert_to_sv_format(frame_with_objects)
            objects_with_track = self.assign_track_id(sv_format)

            # store in dict format instead list format for easy store!

            results[frame_num] = {
                "players" : {},
                "goalkeepers" : {},
                "referees" : {},
                "ball" : {}
            }

            for obj in objects_with_track:
                bbox = obj[0] # xyxy
                cls_id = obj[3]
                track_id = obj[4]

                dict_key = cls_id_to_name[cls_id]
                if cls_id != cls_name_to_id["ball"]:
                    dict_key += "s"
                else: 
                    track_id = 1 # keep ball id alway constant, model maybe lost ball in some frame

                results[frame_num][dict_key][track_id] = {"bbox" : bbox}            

        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(obj=results, file=f)

        return results
    
    def interpolation_ball(self, results):
        
        ball_positions = []
        for frame_num, frame_result in results.items():
            pos = frame_result["ball"].get(1, {}).get("bbox", [])
            ball_positions.append(pos)

        df_ball_pos = pd.DataFrame(data=ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing value
        df_ball_pos = df_ball_pos.interpolate() # default is ?
        

        # If first entry is missing interpolate not process it
        df_ball_pos = df_ball_pos.bfill()

        ball_positions = [bb for bb in df_ball_pos.to_numpy().tolist()]

        for frame_num, frame_result in results.items():
            frame_result["ball"] = {1 : {"bbox" : ball_positions[frame_num]}}

        