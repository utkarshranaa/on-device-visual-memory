"""
Storage module for On-Device Visual Memory.

Handles SQLite database operations for frame metadata.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import DATABASE_PATH


class MetadataStore:
    """
    SQLite-based metadata storage for frame records.

    Schema:
        - id: Primary key
        - filepath: Path to the captured frame image
        - ts: ISO format timestamp
        - indexed: Whether frame has been added to vector index
        - vec_pos: Position in the ANN index (for result mapping)
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the metadata store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or DATABASE_PATH
        self._ensure_database()

    def _get_connection(self):
        """Create and return a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_database(self):
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT NOT NULL,
                ts TEXT NOT NULL,
                indexed INTEGER NOT NULL DEFAULT 0,
                vec_pos INTEGER
            )
        """)

        # Create index on indexed status for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_indexed ON frames(indexed)
        """)

        # Create index on timestamp for time-based filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ts ON frames(ts)
        """)

        conn.commit()
        conn.close()

    def add_frame(self, filepath: str, ts: str) -> int:
        """
        Add a new frame record.

        Args:
            filepath: Path to the frame image file
            ts: ISO format timestamp string

        Returns:
            ID of the inserted record
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO frames (filepath, ts, indexed) VALUES (?, ?, 0)",
            (filepath, ts)
        )

        frame_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return frame_id

    def get_unindexed_frames(self, limit: Optional[int] = None) -> list[dict]:
        """
        Retrieve frames that haven't been indexed yet.

        Args:
            limit: Maximum number of frames to return

        Returns:
            List of frame records as dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM frames WHERE indexed = 0 ORDER BY ts ASC"
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def mark_indexed(self, frame_ids: list[int], vec_pos_start: int = 0):
        """
        Mark frames as indexed and set their vector positions.

        Args:
            frame_ids: List of frame IDs to update
            vec_pos_start: Starting position in the vector index
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        for idx, frame_id in enumerate(frame_ids):
            cursor.execute(
                "UPDATE frames SET indexed = 1, vec_pos = ? WHERE id = ?",
                (vec_pos_start + idx, frame_id)
            )

        conn.commit()
        conn.close()

    def get_frame_by_vec_pos(self, vec_pos: int) -> Optional[dict]:
        """
        Retrieve a frame by its vector index position.

        Args:
            vec_pos: Position in the ANN index

        Returns:
            Frame record as dictionary, or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM frames WHERE vec_pos = ? AND indexed = 1",
            (vec_pos,)
        )

        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def get_frames_by_vec_positions(self, vec_positions: list[int]) -> list[dict]:
        """
        Retrieve multiple frames by their vector index positions.

        Args:
            vec_positions: List of positions in the ANN index

        Returns:
            List of frame records as dictionaries
        """
        if not vec_positions:
            return []

        conn = self._get_connection()
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(vec_positions))
        query = f"SELECT * FROM frames WHERE vec_pos IN ({placeholders}) AND indexed = 1"

        cursor.execute(query, vec_positions)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_frames_since(self, hours: int) -> list[dict]:
        """
        Retrieve frames from the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            List of frame records as dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = datetime.now().timestamp() - (hours * 3600)

        cursor.execute(
            "SELECT * FROM frames WHERE indexed = 1 AND datetime(ts) > datetime(?, 'unixepoch') ORDER BY ts DESC",
            (cutoff,)
        )

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_stats(self) -> dict:
        """
        Get database statistics.

        Returns:
            Dictionary with total_frames, indexed_frames, and unindexed_frames
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM frames")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM frames WHERE indexed = 1")
        indexed = cursor.fetchone()[0]

        conn.close()

        return {
            "total_frames": total,
            "indexed_frames": indexed,
            "unindexed_frames": total - indexed
        }

    def get_all_indexed_paths(self) -> list[str]:
        """
        Get filepaths of all indexed frames.

        Returns:
            List of filepaths
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT filepath FROM frames WHERE indexed = 1 ORDER BY vec_pos")
        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def clear_indexed_flags(self):
        """Reset all indexed flags to 0 (for index rebuild)."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("UPDATE frames SET indexed = 0, vec_pos = NULL")
        conn.commit()
        conn.close()


def create_metadata_store() -> MetadataStore:
    """Factory function to create and return a MetadataStore instance."""
    return MetadataStore()
