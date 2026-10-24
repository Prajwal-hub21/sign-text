import argparse
import csv
import string
from pathlib import Path
from typing import List

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture ASL alphabet samples using MediaPipe hand landmarks."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/samples.csv"),
        help="CSV file where samples will be stored.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Optional path to a video file. Leave empty to use the default webcam.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Flip preview horizontally to mimic a mirror.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum detection confidence required to record a sample.",
    )
    return parser.parse_args()


def normalize_landmarks(landmarks: List[landmark_pb2.NormalizedLandmark]) -> np.ndarray:
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    base = coords[0]
    coords -= base
    norm = np.linalg.norm(coords)
    if norm > 0:
        coords /= norm
    return coords.flatten()


def ensure_header(path: Path, num_features: int) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_names = [f"f_{i:02d}" for i in range(num_features)]
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["label", *feature_names])


def draw_status(frame: np.ndarray, text: str, color=(0, 255, 0)) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    letters = set(string.ascii_uppercase)

    cap = cv2.VideoCapture(0 if not args.video else args.video)
    if not cap.isOpened():
        raise SystemExit("Unable to open video source.")

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence,
        model_complexity=1,
    )
    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    sample_buffer: List[List[float]] = []
    ensure_header(args.output, 63)

    print("Press any alphabet key (A-Z) to save the current hand landmarks.")
    print("Press ESC to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        status = "Show a letter and press its key."
        color = (255, 255, 255)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )
            landmarks = normalize_landmarks(result.multi_hand_landmarks[0].landmark)
            sample_buffer = landmarks.tolist()
            status = "Ready: press the letter you're showing."
            color = (0, 255, 0)
        else:
            sample_buffer = []
            status = "No hand detected."
            color = (0, 0, 255)

        draw_status(frame, status, color=color)
        cv2.imshow("ASL Collector", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
        if not sample_buffer:
            continue

        char = chr(key).upper()
        if char in letters:
            with args.output.open("a", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow([char, *sample_buffer])
            print(f"Saved sample for {char}.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

