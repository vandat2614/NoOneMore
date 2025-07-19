import argparse
from utils import read_video, save_video
from detectors import Detector

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


    print("4. Save result")
    save_video(frames, args.output_path)

if __name__ == '__main__':
    main()
