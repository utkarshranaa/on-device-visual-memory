"""
Streamlit search interface for On-Device Visual Memory.

Provides a web UI to search captured frames using natural language queries.
"""

import io
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VECTOR_INDEX_PATH, EMBEDDING_DIM, DEFAULT_TOP_K
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


class SearchEngine:
    """Handles vector similarity search."""

    def __init__(self):
        """Initialize the search engine with index and metadata."""
        self.store = create_metadata_store()
        self.embedder = None
        self.index = None
        self.backend = None
        self._initialize()

    def _initialize(self):
        """Initialize embedding model and load index."""
        # Load embedding model
        with st.spinner("Loading embedding model..."):
            self.embedder = create_embedding_model()
            device_info = self.embedder.get_device_info()

        # Load vector index
        index_path = VECTOR_INDEX_PATH
        if not index_path.exists():
            st.error(f"Vector index not found at {index_path}. Run build_index.py first!")
            return

        with st.spinner("Loading vector index..."):
            if FAISS_AVAILABLE:
                self.index = faiss.read_index(str(index_path))
                self.backend = "faiss"
            elif HNSWLIB_AVAILABLE:
                self.index = hnswlib.Index(space='ip', dim=self.embedder.dim)
                self.index.load_index(str(index_path))
                self.backend = "hnswlib"
            else:
                st.error("No ANN backend available. Install faiss-cpu or hnswlib.")
                return

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, time_filter: Optional[int] = None) -> list[dict]:
        """
        Search for similar frames.

        Args:
            query: Natural language query
            top_k: Number of results to return
            time_filter: Only return frames from last N hours (None = no filter)

        Returns:
            List of results with metadata and similarity scores
        """
        if self.index is None:
            return []

        # Embed query
        query_embedding = self.embedder.embed_text(query)

        # Search index
        if self.backend == "faiss":
            distances, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), top_k)
            distances = distances[0]
            indices = indices[0]
        else:  # hnswlib
            labels, distances = self.index.knn_query(query_embedding, k=top_k)
            indices = np.array(labels, dtype=np.int64)
            distances = np.array(distances, dtype=np.float32)

        # Fetch metadata
        results = []
        for idx, dist in zip(indices, distances):
            if idx < 0:  # FAISS returns -1 for invalid results
                continue

            frame = self.store.get_frame_by_vec_pos(int(idx))
            if frame is None:
                continue

            # Apply time filter
            if time_filter:
                frame_time = datetime.fromisoformat(frame['ts'])
                if datetime.now() - frame_time > timedelta(hours=time_filter):
                    continue

            results.append({
                'frame': frame,
                'score': float(dist),
                'similarity': float(dist)  # For normalized vectors, inner product = cosine similarity
            })

        return results


def render_result(result: dict):
    """Render a single search result."""
    frame = result['frame']
    score = result['score']
    similarity = result['similarity']

    col1, col2 = st.columns([1, 2])

    with col1:
        try:
            img_path = frame['filepath']
            if Path(img_path).exists():
                img = Image.open(img_path)
                st.image(img, use_container_width=True)
            else:
                st.error("Image not found")
        except Exception as e:
            st.error(f"Error loading image: {e}")

    with col2:
        # Format timestamp
        ts = datetime.fromisoformat(frame['ts'])
        st.caption(f"📅 {ts.strftime('%Y-%m-%d %H:%M:%S')}")

        # Show similarity score
        score_pct = similarity * 100
        delta_color = "normal" if similarity > 0.3 else "inverse"
        st.metric("Similarity", f"{score_pct:.1f}%", delta_color=delta_color)

        # Show filepath
        st.caption(f"📁 {Path(frame['filepath']).name}")

    st.divider()


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="On-Device Visual Memory",
        page_icon="📷",
        layout="wide"
    )

    st.title("On-Device Visual Memory")
    st.caption("Search your camera footage using natural language - entirely offline")

    # Initialize search engine in session state
    if "search_engine" not in st.session_state:
        st.session_state.search_engine = None

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        top_k = st.slider("Results per query", min_value=1, max_value=50, value=DEFAULT_TOP_K)

        time_filter = st.selectbox(
            "Time filter",
            options=[None, 1, 6, 24, 168],  # None, 1h, 6h, 24h, 1 week
            format_func=lambda x: "All time" if x is None else f"Last {x} hours"
        )

        st.divider()

        # Stats section
        st.subheader("📊 Statistics")

        if st.session_state.search_engine is None:
            if st.button("Initialize Search Engine", type="primary"):
                with st.spinner("Initializing..."):
                    st.session_state.search_engine = SearchEngine()
                st.rerun()
        else:
            engine = st.session_state.search_engine
            stats = engine.store.get_stats()

            st.metric("Total frames", stats['total_frames'])
            st.metric("Indexed frames", stats['indexed_frames'])

            if engine.index is not None:
                index_size = engine.index.ntotal if engine.backend == "faiss" else engine.index.get_current_count()
                st.metric("Index size", index_size)

            st.caption(f"Backend: {engine.backend or 'Unknown'}")
            st.caption(f"Dimensions: {EMBEDDING_DIM}")
            st.caption(f"Device: {engine.embedder.get_device_info()['device']}")

        st.divider()

        # Quick example queries
        st.subheader("💡 Example queries")
        examples = [
            "a person sitting",
            "desk with computer",
            "bright room",
            "someone standing",
            "outdoor scene"
        ]
        for example in examples:
            if st.button(example, key=f"example_{example}"):
                st.session_state.query_input = example
                st.rerun()

    # Main search area
    if st.session_state.search_engine is None:
        st.info("👈 Click 'Initialize Search Engine' in the sidebar to start")
        return

    engine = st.session_state.search_engine

    if engine.index is None:
        st.error("Search engine not initialized properly")
        return

    # Search input
    query = st.text_input(
        "🔍 Search your footage",
        value=st.session_state.get("query_input", ""),
        placeholder="Describe what you're looking for..."
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("Search", type="primary", use_container_width=True)

    # Perform search
    if search_button and query.strip():
        with st.spinner(f"Searching for '{query}'..."):
            results = engine.search(
                query=query.strip(),
                top_k=top_k,
                time_filter=time_filter
            )

        # Display results
        st.subheader(f"Results ({len(results)} found)")

        if not results:
            st.info("No matching frames found. Try a different query or adjust the time filter.")
        else:
            for i, result in enumerate(results, 1):
                st.caption(f"**Result {i}**")
                render_result(result)

    # Clear query from session state after displaying
    if "query_input" in st.session_state:
        del st.session_state.query_input


if __name__ == "__main__":
    main()
