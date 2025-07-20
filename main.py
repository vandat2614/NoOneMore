import argparse
from utils import read_video, save_video, draw
from detectors import Detector, Assigner

def parse_args():
    parser = argparse.ArgumentParser(description="Read a video and save it after processing.")
    parser.add_argument("--input_path", "-i", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output_path", "-o", type=str, required=True, help="Path to save the output video")
    parser.add_argument("--weight", type=str, required=True, help="path to detection model weight")
    parser.add_argument("--save_path", type=str, required=True, help="path to processed checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()

    print("1. Start load video")
    frames = read_video(args.input_path)
    
    print("2. Detect Object")
    
    detector = Detector(weight_path=args.weight)
    print("2.1 Load weight successful")
    
    results = detector.process(frames, load=True, save_path=args.save_path)

    print("2.2 Detect done")

    print("2.3 Interpolate ball")
    detector.interpolation_ball(results)



    print("3. Assign Team")
    assigner = Assigner()
    assigner.fit(frames[0], results[0]["players"])
    
    for frame_num, frame in enumerate(frames):
        objects = results[frame_num]["players"]

        for player_id, info in objects.items():
            team = assigner.predict(frame, info["bbox"], player_id)

            results[frame_num]["players"][player_id]["team"] = team
            results[frame_num]["players"][player_id]["team_color"] = assigner.team1_color if team == 1 else assigner.team2_color

        player_keep_ball_id = assigner.assign_ball_to_player(objects, results[frame_num]["ball"][1]["bbox"])

        if player_keep_ball_id is not None:
            results[frame_num]["players"][player_keep_ball_id]["has_ball"] = True


    print("4. Draw")
    output_frames = draw(frames, results)

    print("5. Save result")
    save_video(output_frames, args.output_path)

if __name__ == '__main__':
    main()
