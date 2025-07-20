import cv2
import os
import numpy as np
from .bbox_utils import get_center_of_bbox, get_bbox_width

def read_video(video_path):
    cap = cv2.VideoCapture(filename=video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(frames, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename=output_path, 
                          fourcc=fourcc, 
                          fps=24,
                          frameSize=(frames[0].shape[1], frames[0].shape[0])) # width , height
    for frame in frames:
        out.write(frame)
    out.release() 

def draw_ellipse(frame, bbox, track_id, color=(0,0,0)):
    y2 = bbox[3]
    x_center, y_center = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(frame,
                center=(int(x_center), int(y2)),
                axes=(int(width), int(0.35 * width)),# radius of (minor axis, major axis)
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=color,
                thickness=2,
                lineType=cv2.LINE_4) 
    
    rectangle_width = 40
    rectangle_height = 20
    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = (y2 - rectangle_height // 2) + 15
    y2_rect = (y2 + rectangle_height // 2) + 15

    cv2.rectangle(frame,
                    (int(x1_rect), int(y1_rect)),
                    (int(x2_rect), int(y2_rect)),
                    color,
                    cv2.FILLED)
    
    x1_text = x1_rect + 12
    if track_id > 99:
        x1_text -= 10
    
    cv2.putText(frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 0, 0),
                thickness=2)

    return frame

def draw_triangle(frame, bbox, color):
    y = bbox[1]
    x, _ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x, y], #bottom
        [x-10, y-20], # left top
        [x+10, y-20] # right top
    ], dtype=np.int32).reshape((-1, 1, 2))

    # 0 is contours index (>=0 vẽ contour tại index đó), = -1 thì vẽ all trong lisst (hiện tại chỉ có 1 )
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2) # border

    return frame

def draw(frames, tracks):

    output_frames = []
    for frame_num, frame in enumerate(frames):
        draw_frame = frame.copy()

        player_dict = tracks[frame_num]["players"]
        goalkeeper_dict = tracks[frame_num]["goalkeepers"]
        ball_dict = tracks[frame_num]["ball"]
        referee_dict = tracks[frame_num]["referees"]

        for track_id, info in player_dict.items():
            draw_frame = draw_ellipse(draw_frame, info["bbox"], track_id, color=info["team_color"])

            if info.get("has_ball", False):
                draw_frame = draw_triangle(draw_frame, info["bbox"], (255, 0, 255))
        
        for track_id, info in referee_dict.items():
            draw_frame = draw_ellipse(draw_frame, info["bbox"], track_id, color=(0,255,255))

        for track_id, info in goalkeeper_dict.items():
            draw_frame = draw_ellipse(draw_frame, info["bbox"], track_id, color=(0, 255, 0))

        for track_id, info in ball_dict.items():
            draw_frame = draw_triangle(draw_frame, info["bbox"], (255, 0, 0))

        output_frames.append(draw_frame)

    return output_frames