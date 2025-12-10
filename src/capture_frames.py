"""
Camera frame capture module for On-Device Visual Memory.

Captures frames from a webcam at regular intervals and stores metadata.
"""

import argparse
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from config import (
    FRAMES_DIR,
    DEFAULT_CAMERA_DEVICE,
    DEFAULT_CAPTURE_INTERVAL,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_MAX_FRAMES,
)
from store import create_metadata_store


class FrameCapture:
    """
    Captures frames from a camera at regular intervals.

    Saves frames as JPEG files and records metadata in SQLite.
    """

    def __init__(
        self,
        device: int = DEFAULT_CAMERA_DEVICE,
        interval: float = DEFAULT_CAPTURE_INTERVAL,
        width: int = DEFAULT_FRAME_WIDTH,
        height: int = DEFAULT_FRAME_HEIGHT,
        max_frames: Optional[int] = DEFAULT_MAX_FRAMES,
    ):
        """
        Initialize the frame capture.

        Args:
            device: Camera device ID (usually 0)
            interval: Seconds between frame captures
            width: Frame width in pixels
            height: Frame height in pixels
            max_frames: Maximum number of frames to capture (None for unlimited)
        """
        self.device = device
        self.interval = interval
        self.width = width
        self.height = height
        self.max_frames = max_frames
        self.frame_count = 0
        self.running = False

        # Initialize metadata store
        self.store = create_metadata_store()

        # Initialize camera
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera device {device}")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Get actual dimensions (may differ from requested)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📷 Camera initialized: {actual_width}x{actual_height}")

    def capture_frame(self) -> Optional[str]:
        """
        Capture a single frame and save it.

        Returns:
            Path to the saved frame file, or None if capture failed
        """
        ret, frame = self.cap.read()
        if not ret:
            print("⚠️  Failed to capture frame")
            return None

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}.jpg"
        filepath = FRAMES_DIR / filename

        # Save frame
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Store metadata
        iso_timestamp = datetime.now().isoformat()
        frame_id = self.store.add_frame(str(filepath), iso_timestamp)

        self.frame_count += 1

        return str(filepath)

    def start(self):
        """Start continuous capture loop."""
        self.running = True
        print(f"🎬 Starting capture (interval: {self.interval}s, max frames: {self.max_frames or 'unlimited'})")
        print("Press Ctrl+C to stop")

        import time

        try:
            while self.running:
                filepath = self.capture_frame()
                if filepath:
                    print(f"✓ Captured frame {self.frame_count}: {Path(filepath).name}")

                # Check max frames
                if self.max_frames and self.frame_count >= self.max_frames:
                    print(f"🏁 Reached maximum frame count ({self.max_frames})")
                    break

                # Wait for next interval
                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n⏹️  Capture stopped by user")

        finally:
            self.stop()

    def stop(self):
        """Stop capture and release resources."""
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
            cv2.destroyAllWindows()
        print(f"📊 Total frames captured: {self.frame_count}")


def main():
    """CLI entry point for capture_frames.py"""
    parser = argparse.ArgumentParser(
        description="Capture camera frames at regular intervals"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=DEFAULT_CAMERA_DEVICE,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_CAPTURE_INTERVAL,
        help="Seconds between frame captures (default: 2.0)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help="Frame width in pixels (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_FRAME_HEIGHT,
        help="Frame height in pixels (default: 720)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to capture (default: unlimited)"
    )

    args = parser.parse_args()

    try:
        capture = FrameCapture(
            device=args.device,
            interval=args.interval,
            width=args.width,
            height=args.height,
            max_frames=args.max_frames,
        )
        capture.start()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
