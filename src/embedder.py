"""
Embedding module for On-Device Visual Memory.

Provides CLIP-based embeddings for images and text using open_clip_torch.
Supports CUDA acceleration when available.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from PIL import Image
import open_clip

from config import CLIP_MODEL_NAME, CLIP_PRETRAINED, EMBEDDING_DIM


class EmbeddingModel:
    """
    CLIP-based embedding model for images and text.

    Automatically uses CUDA if available, otherwise falls back to CPU.
    All embeddings are normalized to unit length for cosine similarity.
    """

    def __init__(self, model_name: str = CLIP_MODEL_NAME, pretrained: str = CLIP_PRETRAINED):
        """
        Initialize the CLIP model.

        Args:
            model_name: Name of the CLIP model architecture
            pretrained: Pretrained weights to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.pretrained = pretrained

        # Load model and preprocessing
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device
        )
        self.model.eval()

        # Get embedding dimension
        self.dim = EMBEDDING_DIM

    def embed_image(self, image: Image.Image | np.ndarray | str) -> np.ndarray:
        """
        Generate embedding for a single image.

        Args:
            image: PIL Image, numpy array (BGR), or file path string

        Returns:
            Normalized embedding vector as float32 numpy array
        """
        # Handle different input types
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Assume BGR from OpenCV, convert to RGB
            if image.ndim == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]  # BGR to RGB
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Preprocess and generate embedding
        with torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_input)
            embedding = image_features.cpu().numpy()[0]

        # Normalize to unit length
        return self._normalize(embedding)

    def embed_images(self, images: list) -> np.ndarray:
        """
        Generate embeddings for multiple images in batch.

        Args:
            images: List of PIL Images, numpy arrays, or file paths

        Returns:
            Matrix of normalized embeddings (N x D)
        """
        embeddings = []
        for img in images:
            embeddings.append(self.embed_image(img))
        return np.stack(embeddings, axis=0)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text query.

        Args:
            text: Natural language query string

        Returns:
            Normalized embedding vector as float32 numpy array
        """
        # Tokenize and generate embedding
        with torch.no_grad():
            text_input = open_clip.tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text_input)
            embedding = text_features.cpu().numpy()[0]

        # Normalize to unit length
        return self._normalize(embedding)

    def embed_text_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for multiple text queries.

        Args:
            texts: List of query strings

        Returns:
            Matrix of normalized embeddings (N x D)
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))
        return np.stack(embeddings, axis=0)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """
        L2 normalize a vector to unit length.

        Args:
            vector: Input vector

        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return (vector / norm).astype(np.float32)
        return vector.astype(np.float32)

    def get_device_info(self) -> dict:
        """Return information about the compute device."""
        return {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "model": self.model_name,
            "embedding_dim": self.dim,
        }


def create_embedding_model() -> EmbeddingModel:
    """Factory function to create and return an EmbeddingModel instance."""
    return EmbeddingModel()
