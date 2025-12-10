"""
Vector index builder for On-Device Visual Memory.

Supports FAISS (preferred) with automatic fallback to hnswlib.
Handles incremental index building from unindexed frames.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    VECTOR_INDEX_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_NEW_FRAMES,
    EMBEDDING_DIM,
)
from embedder import create_embedding_model
from store import create_metadata_store


# ANN backend detection
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False


class ANNIndex:
    """
    Abstract wrapper for ANN vector index supporting FAISS and hnswlib.

    Automatically selects FAISS if available, otherwise falls back to hnswlib.
    """

    def __init__(self, dim: int, index_path: Path, backend: Optional[str] = None):
        """
        Initialize the ANN index.

        Args:
            dim: Embedding dimension
            index_path: Path to save/load the index file
            backend: Force specific backend ('faiss' or 'hnswlib'), auto-detect if None
        """
        self.dim = dim
        self.index_path = index_path
        self.backend = self._select_backend(backend)
        self.index = None
        self.count = 0

    def _select_backend(self, backend: Optional[str]) -> str:
        """Select the ANN backend based on availability."""
        if backend:
            if backend == "faiss" and not FAISS_AVAILABLE:
                raise RuntimeError("FAISS requested but not available")
            if backend == "hnswlib" and not HNSWLIB_AVAILABLE:
                raise RuntimeError("hnswlib requested but not available")
            return backend

        if FAISS_AVAILABLE:
            return "faiss"
        elif HNSWLIB_AVAILABLE:
            return "hnswlib"
        else:
            raise RuntimeError(
                "No ANN backend available. Install faiss-cpu or hnswlib:\n"
                "  pip install faiss-cpu\n"
                "  pip install hnswlib"
            )

    def create_index(self):
        """Create a new empty index."""
        if self.backend == "faiss":
            self.index = faiss.IndexFlatIP(self.dim)  # Inner product for normalized vectors
        else:  # hnswlib
            self.index = hnswlib.Index(space='ip', dim=self.dim)
            self.index.init_index(max_elements=1000000, ef_construction=200, M=16)
        self.count = 0

    def load_index(self) -> bool:
        """
        Load existing index from disk.

        Returns:
            True if index was loaded, False if file doesn't exist
        """
        if not self.index_path.exists():
            return False

        if self.backend == "faiss":
            self.index = faiss.read_index(str(self.index_path))
            self.count = self.index.ntotal
        else:  # hnswlib
            self.index = hnswlib.Index(space='ip', dim=self.dim)
            self.index.load_index(str(self.index_path))
            self.count = self.index.get_current_count()

        return True

    def save_index(self):
        """Save index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self.backend == "faiss":
            faiss.write_index(self.index, str(self.index_path))
        else:  # hnswlib
            self.index.save_index(str(self.index_path))

    def add_vectors(self, vectors: np.ndarray):
        """
        Add vectors to the index.

        Args:
            vectors: Array of shape (N, D) with normalized embeddings
        """
        n = vectors.shape[0]

        if self.backend == "faiss":
            self.index.add(vectors)
        else:  # hnswlib
            # hnswlib requires ids starting from current count
            ids = np.arange(self.count, self.count + n).astype(np.int64)
            self.index.add_items(vectors, ids)

        self.count += n

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.

        Args:
            query: Query vector of shape (D,)
            k: Number of results to return

        Returns:
            Tuple of (distances, indices) - arrays of length k
        """
        if self.backend == "faiss":
            distances, indices = self.index.search(query.reshape(1, -1).astype('float32'), k)
            return distances[0], indices[0]
        else:  # hnswlib
            labels, distances = self.index.knn_query(query, k=k)
            return np.array(distances, dtype=np.float32), np.array(labels, dtype=np.int64)

    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self.count


def build_incremental_index(
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_new: Optional[int] = DEFAULT_MAX_NEW_FRAMES,
    index_path: Optional[Path] = None
):
    """
    Build or update the vector index incrementally.

    Only processes frames that haven't been indexed yet (indexed=0 in database).
    """
    index_path = index_path or VECTOR_INDEX_PATH
    print(f"🔍 Starting incremental index build...")
    print(f"📁 Index path: {index_path}")

    # Initialize components
    store = create_metadata_store()
    stats = store.get_stats()
    print(f"📊 Database stats: {stats['total_frames']} total, {stats['indexed_frames']} indexed, {stats['unindexed_frames']} unindexed")

    if stats['unindexed_frames'] == 0:
        print("✅ No new frames to index!")
        return

    # Get unindexed frames
    limit = max_new if max_new else None
    unindexed = store.get_unindexed_frames(limit=limit)
    print(f"📋 Processing {len(unindexed)} unindexed frames")

    # Initialize embedding model
    print("🧠 Loading embedding model...")
    embedder = create_embedding_model()
    device_info = embedder.get_device_info()
    print(f"   Device: {device_info['device']}")
    print(f"   Model: {device_info['model']}")
    print(f"   Dimensions: {device_info['embedding_dim']}")

    # Initialize or load index
    print(f"📦 Initializing ANN index...")
    ann_index = ANNIndex(dim=embedder.dim, index_path=index_path)

    if ann_index.load_index():
        print(f"   Loaded existing index with {ann_index.size} vectors")
    else:
        ann_index.create_index()
        print(f"   Created new index")

    # Process in batches
    from PIL import Image

    total_to_process = len(unindexed)
    processed = 0
    batch_embeddings = []
    batch_ids = []

    for frame in unindexed:
        try:
            # Load and embed image
            img_path = frame['filepath']
            if not Path(img_path).exists():
                print(f"⚠️  Skipping missing file: {img_path}")
                continue

            img = Image.open(img_path).convert("RGB")
            embedding = embedder.embed_image(img)
            batch_embeddings.append(embedding)
            batch_ids.append(frame['id'])

            # Process batch when full
            if len(batch_embeddings) >= batch_size:
                embeddings_matrix = np.stack(batch_embeddings)
                ann_index.add_vectors(embeddings_matrix)

                # Get starting vector position
                vec_pos_start = ann_index.size - len(batch_ids)
                store.mark_indexed(batch_ids, vec_pos_start)

                processed += len(batch_ids)
                print(f"   Progress: {processed}/{total_to_process} frames")

                batch_embeddings = []
                batch_ids = []

        except Exception as e:
            print(f"⚠️  Error processing frame {frame['id']}: {e}")

    # Process remaining batch
    if batch_embeddings:
        embeddings_matrix = np.stack(batch_embeddings)
        ann_index.add_vectors(embeddings_matrix)
        vec_pos_start = ann_index.size - len(batch_ids)
        store.mark_indexed(batch_ids, vec_pos_start)
        processed += len(batch_ids)

    # Save index
    print(f"💾 Saving index with {ann_index.size} total vectors...")
    ann_index.save_index()

    print(f"✅ Index build complete! Backend: {ann_index.backend}")


def main():
    """CLI entry point for build_index.py"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build or update the vector index for captured frames"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of frames to process per batch (default: 32)"
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="Maximum number of new frames to index (default: all)"
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Path to vector index file"
    )

    args = parser.parse_args()

    build_incremental_index(
        batch_size=args.batch_size,
        max_new=args.max_new,
        index_path=args.index_path
    )


if __name__ == "__main__":
    main()
