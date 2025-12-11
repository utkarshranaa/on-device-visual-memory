"""
Benchmark module for On-Device Visual Memory.

Measures embedding performance, search latency, and resource usage.
"""

import gc
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from config import VECTOR_INDEX_PATH, EMBEDDING_DIM
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

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class Benchmark:
    """Benchmark suite for Visual Memory performance."""

    def __init__(self):
        """Initialize benchmark components."""
        self.embedder = create_embedding_model()
        self.store = create_metadata_store()
        self.index = None
        self.backend = None
        self._load_index()

    def _load_index(self):
        """Load the vector index."""
        index_path = VECTOR_INDEX_PATH
        if not index_path.exists():
            print(f"⚠️  Vector index not found at {index_path}")
            return

        if FAISS_AVAILABLE:
            self.index = faiss.read_index(str(index_path))
            self.backend = "faiss"
        elif HNSWLIB_AVAILABLE:
            self.index = hnswlib.Index(space='ip', dim=self.embedder.dim)
            self.index.load_index(str(index_path))
            self.backend = "hnswlib"

    def benchmark_embedding(self, sample_size: int = 50) -> dict:
        """
        Benchmark image embedding performance.

        Args:
            sample_size: Number of frames to test

        Returns:
            Dictionary with timing statistics
        """
        print(f"\n🧪 Benchmarking embedding performance (n={sample_size})...")

        # Get sample of indexed frames
        indexed_paths = self.store.get_all_indexed_paths()
        if not indexed_paths:
            print("   No indexed frames found for benchmarking")
            return {}

        sample_size = min(sample_size, len(indexed_paths))
        sample_paths = indexed_paths[:sample_size]

        times = []
        for path in sample_paths:
            try:
                img = Image.open(path).convert("RGB")

                start = time.perf_counter()
                _ = self.embedder.embed_image(img)
                end = time.perf_counter()

                times.append((end - start) * 1000)  # Convert to ms

            except Exception as e:
                print(f"   ⚠️  Error processing {Path(path).name}: {e}")

        if not times:
            return {}

        times_array = np.array(times)
        return {
            "count": len(times),
            "mean_ms": float(np.mean(times_array)),
            "std_ms": float(np.std(times_array)),
            "p50_ms": float(np.percentile(times_array, 50)),
            "p95_ms": float(np.percentile(times_array, 95)),
            "p99_ms": float(np.percentile(times_array, 99)),
            "min_ms": float(np.min(times_array)),
            "max_ms": float(np.max(times_array)),
        }

    def benchmark_search(self, num_queries: int = 50, top_k: int = 10) -> dict:
        """
        Benchmark search latency.

        Args:
            num_queries: Number of test queries to run
            top_k: Number of results per query

        Returns:
            Dictionary with timing statistics
        """
        print(f"\n🔍 Benchmarking search performance (queries={num_queries}, k={top_k})...")

        if self.index is None:
            print("   ⚠️  No index available for search benchmark")
            return {}

        # Built-in test queries
        test_queries = [
            "a person sitting down",
            "desk with computer monitor",
            "bright room with windows",
            "someone standing up",
            "outdoor scene with trees",
            "kitchen with appliances",
            "living room with furniture",
            "office workspace",
            "bookshelf with books",
            "night time indoor scene",
            "hand holding object",
            "screen display",
            "door or entrance",
            "wall with decorations",
            "table surface",
            "chair or seating",
            "electronics device",
            "window with light",
            "floor with carpet",
            "ceiling with lights",
        ]

        # Extend queries if needed
        while len(test_queries) < num_queries:
            test_queries.extend(test_queries)
        test_queries = test_queries[:num_queries]

        times = []
        for query in test_queries:
            start = time.perf_counter()

            query_embedding = self.embedder.embed_text(query)

            if self.backend == "faiss":
                self.index.search(query_embedding.reshape(1, -1).astype('float32'), top_k)
            else:  # hnswlib
                self.index.knn_query(query_embedding, k=top_k)

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        times_array = np.array(times)
        return {
            "count": len(times),
            "mean_ms": float(np.mean(times_array)),
            "std_ms": float(np.std(times_array)),
            "p50_ms": float(np.percentile(times_array, 50)),
            "p95_ms": float(np.percentile(times_array, 95)),
            "p99_ms": float(np.percentile(times_array, 99)),
            "min_ms": float(np.min(times_array)),
            "max_ms": float(np.max(times_array)),
        }

    def get_memory_usage(self) -> dict:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with memory info
        """
        if not PSUTIL_AVAILABLE:
            return {
                "available": False,
                "note": "Install psutil for memory measurements: pip install psutil"
            }

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "available": True,
            "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size
            "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size
        }

    def run_full_benchmark(self, embedding_samples: int = 50, search_queries: int = 50) -> dict:
        """
        Run complete benchmark suite.

        Args:
            embedding_samples: Number of frames for embedding benchmark
            search_queries: Number of queries for search benchmark

        Returns:
            Complete benchmark results
        """
        print("=" * 60)
        print("ON-DEVICE VISUAL MEMORY - BENCHMARK")
        print("=" * 60)

        # System info
        device_info = self.embedder.get_device_info()
        print(f"\n🖥️  Device: {device_info['device']}")
        print(f"📦 Model: {device_info['model']}")
        print(f"📐 Dimensions: {device_info['embedding_dim']}")
        print(f"🔧 Index backend: {self.backend or 'None'}")

        # Database stats
        stats = self.store.get_stats()
        print(f"\n📊 Database: {stats['total_frames']} total, {stats['indexed_frames']} indexed")

        # Run benchmarks
        results = {
            "timestamp": datetime.now().isoformat(),
            "device_info": device_info,
            "database_stats": stats,
            "index_backend": self.backend,
        }

        # Embedding benchmark
        if stats['indexed_frames'] > 0:
            results["embedding"] = self.benchmark_embedding(sample_size=embedding_samples)

        # Search benchmark
        if self.index is not None:
            results["search"] = self.benchmark_search(num_queries=search_queries)

        # Memory usage
        results["memory"] = self.get_memory_usage()

        return results

    def print_results(self, results: dict):
        """Print benchmark results in a formatted way."""
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        if "embedding" in results and results["embedding"]:
            emb = results["embedding"]
            print(f"\n📸 Image Embedding:")
            print(f"   Samples:    {emb['count']}")
            print(f"   Mean:       {emb['mean_ms']:.2f} ms")
            print(f"   P50:        {emb['p50_ms']:.2f} ms")
            print(f"   P95:        {emb['p95_ms']:.2f} ms")
            print(f"   P99:        {emb['p99_ms']:.2f} ms")

        if "search" in results and results["search"]:
            srch = results["search"]
            print(f"\n🔍 Search Latency:")
            print(f"   Queries:    {srch['count']}")
            print(f"   Mean:       {srch['mean_ms']:.2f} ms")
            print(f"   P50:        {srch['p50_ms']:.2f} ms")
            print(f"   P95:        {srch['p95_ms']:.2f} ms")
            print(f"   P99:        {srch['p99_ms']:.2f} ms")

        if "memory" in results:
            mem = results["memory"]
            print(f"\n💾 Memory Usage:")
            if mem.get("available"):
                print(f"   RSS:        {mem['rss_mb']:.2f} MB")
                print(f"   VMS:        {mem['vms_mb']:.2f} MB")
            else:
                print(f"   {mem.get('note', 'Not available')}")

        print("\n" + "=" * 60)


def main():
    """CLI entry point for benchmark.py"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Visual Memory performance"
    )
    parser.add_argument(
        "--embedding-samples",
        type=int,
        default=50,
        help="Number of frames for embedding benchmark (default: 50)"
    )
    parser.add_argument(
        "--search-queries",
        type=int,
        default=50,
        help="Number of queries for search benchmark (default: 50)"
    )

    args = parser.parse_args()

    benchmark = Benchmark()
    results = benchmark.run_full_benchmark(
        embedding_samples=args.embedding_samples,
        search_queries=args.search_queries
    )
    benchmark.print_results(results)


if __name__ == "__main__":
    main()
