import argparse
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import cv2
import joblib
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import tkinter as tk
from tkinter import messagebox, scrolledtext

from storage import Storage
from grammar import GrammarHelper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time sign-to-text recognizer.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/alphabet_clf.joblib"),
        help="Model bundle produced by train_alphabet.py",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum probability required to accept a prediction.",
    )
    parser.add_argument(
        "--stability",
        type=int,
        default=3,
        help="Number of consecutive frames with the same prediction before committing.",
    )
    parser.add_argument(
        "--instant-confidence",
        type=float,
        default=0.9,
        help="Confidence threshold to bypass stability check for fast commits.",
    )
    parser.add_argument(
        "--repeat-delay",
        type=float,
        default=0.6,
        help="Seconds to wait before repeating the same letter automatically.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Flip preview horizontally to mimic a mirror.",
    )
    return parser.parse_args()


def normalize_landmarks(
    landmarks: list[landmark_pb2.NormalizedLandmark],
) -> np.ndarray:
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    base = coords[0]
    coords -= base
    norm = np.linalg.norm(coords)
    if norm > 0:
        coords /= norm
    return coords.flatten()


class RecognizerApp:
    def __init__(self, args: argparse.Namespace) -> None:
        if not args.model.exists():
            raise SystemExit(f"Model not found: {args.model}. Train it first.")
        bundle = joblib.load(args.model)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.class_labels = list(self.model.classes_)

        self.args = args
        self.storage = Storage()
        self.grammar_helper = GrammarHelper()
        self.running = True
        self.last_prediction: Optional[str] = None
        self.last_confidence: float = 0.0
        self.last_commit_time: float = 0.0
        self.prediction_queue: Deque[str] = deque(maxlen=args.stability)
        self.last_corrected_text: Optional[str] = None

        self.root = tk.Tk()
        self.root.title("Sign Language to Text")
        self.text_value = tk.StringVar(self.root, value="")
        self._build_ui()
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="Recognized Text").pack(anchor="w")
        self.text_widget = scrolledtext.ScrolledText(frame, height=4, width=40, font=("Arial", 16))
        self.text_widget.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.text_widget.insert(tk.END, "")

        tk.Label(frame, text="AI Grammar Suggestion").pack(anchor="w")
        self.grammar_widget = scrolledtext.ScrolledText(
            frame, height=4, width=40, font=("Arial", 14), state=tk.DISABLED
        )
        self.grammar_widget.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        button_frame = tk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=5)

        tk.Button(button_frame, text="Space", command=lambda: self.append_character(" ")).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )
        tk.Button(button_frame, text="Delete", command=self.delete_last).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )
        tk.Button(button_frame, text="Clear All", command=self.clear_text).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )

        grammar_frame = tk.Frame(frame)
        grammar_frame.pack(fill=tk.X, pady=5)
        self.grammar_btn = tk.Button(
            grammar_frame,
            text="Grammar Fix",
            command=self.run_grammar_fix,
            state=tk.NORMAL if self.grammar_helper.available else tk.DISABLED,
        )
        self.grammar_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(
            grammar_frame,
            text="Apply Suggestion",
            command=self.apply_grammar_suggestion,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        tk.Button(frame, text="Save to DB", command=self.save_sentence, bg="#4CAF50", fg="white").pack(
            fill=tk.X, pady=(10, 5)
        )
        tk.Button(frame, text="Quit", command=self.on_quit, bg="#F44336", fg="white").pack(fill=tk.X)

        info = (
            "Tips:\n"
            "- Hold a letter steady for a moment to record it.\n"
            "- Use Space/Delete buttons to fix mistakes.\n"
            "- Press 'Grammar Fix' to get AI suggestions.\n"
            "- Press 'Save to DB' to store the sentence."
        )
        tk.Label(frame, text=info, justify="left", fg="#555").pack(anchor="w", pady=(10, 0))

    def _can_append(self, char: str) -> bool:
        """Restrict consecutive repeats to at most two identical characters."""
        content = self.text_widget.get("1.0", tk.END).rstrip("\n")
        if len(content) >= 2 and content[-1] == char and content[-2] == char:
            return False
        return True

    def append_character(self, char: str, confidence: float | None = None) -> None:
        if not self._can_append(char):
            return
        self.text_widget.insert(tk.END, char)
        self.text_widget.see(tk.END)
        self.storage.log_letter(char, confidence)

    def delete_last(self) -> None:
        content = self.text_widget.get("1.0", tk.END).rstrip("\n")
        if not content:
            return
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, content[:-1])

    def clear_text(self) -> None:
        self.text_widget.delete("1.0", tk.END)

    def get_current_text(self) -> str:
        return self.text_widget.get("1.0", tk.END).strip()

    def run_grammar_fix(self) -> None:
        if not self.grammar_helper.available:
            messagebox.showerror("Grammar Fix", "Grammar engine not available. Install Java + language-tool-python.")
            return
        content = self.get_current_text()
        if not content:
            messagebox.showinfo("Grammar Fix", "Enter text first.")
            return
        result = self.grammar_helper.correct(content)
        self.last_corrected_text = result.corrected
        self.grammar_widget.config(state=tk.NORMAL)
        self.grammar_widget.delete("1.0", tk.END)
        self.grammar_widget.insert(tk.END, result.corrected)
        self.grammar_widget.config(state=tk.DISABLED)
        messagebox.showinfo(
            "Grammar Fix",
            f"Suggested text prepared (issues found: {result.issues_found}). Click 'Apply Suggestion' to use it.",
        )

    def apply_grammar_suggestion(self) -> None:
        if not self.last_corrected_text:
            messagebox.showinfo("Grammar Fix", "No suggestion yet. Run 'Grammar Fix' first.")
            return
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, self.last_corrected_text)
        self.text_widget.see(tk.END)

    def save_sentence(self) -> None:
        content = self.text_widget.get("1.0", tk.END).strip()
        if not content:
            messagebox.showinfo("Save Sentence", "Nothing to save yet.")
            return
        sentence_id = self.storage.save_sentence(content)
        messagebox.showinfo("Save Sentence", f"Saved sentence #{sentence_id}")
        self.clear_text()

    def on_quit(self) -> None:
        self.running = False
        self.root.after(100, self._shutdown)

    def _shutdown(self) -> None:
        self.storage.close()
        cv2.destroyAllWindows()
        self.root.quit()

    def _video_loop(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to open webcam.")
            self.on_quit()
            return

        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1,
        )
        drawing = mp.solutions.drawing_utils
        drawing_styles = mp.solutions.drawing_styles

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            if self.args.mirror:
                frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            overlay_text = "No hand detected"
            overlay_color = (0, 0, 255)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style(),
                    )
                vector = normalize_landmarks(result.multi_hand_landmarks[0].landmark)
                scaled = self.scaler.transform([vector])
                probs = self.model.predict_proba(scaled)[0]
                idx = int(np.argmax(probs))
                pred_letter = self.class_labels[idx]
                confidence = float(probs[idx])
                overlay_text = f"{pred_letter} ({confidence:.2f})"
                overlay_color = (0, 255, 0)

                self.prediction_queue.append(pred_letter if confidence >= self.args.min_confidence else "")

                stable = (
                    len(self.prediction_queue) == self.prediction_queue.maxlen
                    and self.prediction_queue.count(pred_letter) == self.prediction_queue.maxlen
                    and confidence >= self.args.min_confidence
                )
                confident_enough = confidence >= self.args.instant_confidence
                cooldown_ok = (time.monotonic() - self.last_commit_time) >= self.args.repeat_delay

                if (stable or confident_enough) and (pred_letter != self.last_prediction or cooldown_ok):
                    self.root.after(
                        0, lambda letter=pred_letter, conf=confidence: self.append_character(letter, conf)
                    )
                    self.last_prediction = pred_letter
                    self.last_confidence = confidence
                    self.last_commit_time = time.monotonic()
                    self.prediction_queue.clear()
                elif pred_letter != self.last_prediction and stable:
                    self.prediction_queue.clear()
            else:
                self.prediction_queue.clear()
                self.last_prediction = None

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(
                frame,
                overlay_text,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                overlay_color,
                2,
            )
            cv2.imshow("Sign Language Feed", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                self.root.after(0, self.on_quit)
                break

        cap.release()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    app = RecognizerApp(args)
    app.run()


if __name__ == "__main__":
    main()

