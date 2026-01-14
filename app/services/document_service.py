
from __future__ import annotations

import logging
import os
from typing import Optional

from langchain_aws import BedrockEmbeddings as LangchainAwsBedrockEmbeddings
from langchain_community.embeddings import BedrockEmbeddings as LangchainCommunityBedrockEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentServiceError(RuntimeError):
	pass


class DocumentService:
	def __init__(
		self,
		*,
		embedding_model_id: Optional[str] = None,
		embedding_dim: Optional[int] = None,
	) -> None:
		self._embedding_model_id = (embedding_model_id or os.getenv("BEDROCK_EMBEDDING_MODEL_ID") or "").strip()
		self._embedding_dim = embedding_dim or self._read_embedding_dim_from_env()

	@staticmethod
	def _read_embedding_dim_from_env() -> int:
		raw = (os.getenv("BEDROCK_EMBEDDING_DIM") or "").strip()
		if not raw:
			return 1024
		try:
			return int(raw)
		except ValueError as exc:
			raise ValueError("BEDROCK_EMBEDDING_DIM must be an int") from exc

	def embed_text(self, *, text: str) -> list[float]:
		"""Embed a single text string using Amazon Bedrock via LangChain.

		Returns:
			A list[float] embedding.
		"""

		if text is None or not str(text).strip():
			raise ValueError("text must be provided")

		try:
			embeddings = self._get_bedrock_embeddings_client()
			vector = embeddings.embed_query(text)
		except Exception as exc:
			logger.exception("Embedding generation failed")
			raise DocumentServiceError("Failed to generate embedding") from exc

		if not isinstance(vector, list) or not all(isinstance(x, (int, float)) for x in vector):
			raise DocumentServiceError("Unexpected embedding output type")

		if len(vector) != self._embedding_dim:
			raise DocumentServiceError(
				f"Embedding dimension mismatch (expected {self._embedding_dim}, got {len(vector)})"
			)

		return [float(x) for x in vector]

	def chunk_text(
		self,
		*,
		text: str,
		chunk_size: int = 500,
		chunk_overlap: int = 50,
	) -> list[str]:
		"""Split text into overlapping chunks using LangChain.

		Args:
			text: Input text.
			chunk_size: Target chunk size in characters.
			chunk_overlap: Overlap size in characters.
		"""

		if text is None or not str(text).strip():
			return []
		if chunk_size <= 0:
			raise ValueError("chunk_size must be > 0")
		if chunk_overlap < 0:
			raise ValueError("chunk_overlap must be >= 0")
		if chunk_overlap >= chunk_size:
			raise ValueError("chunk_overlap must be < chunk_size")

		splitter = RecursiveCharacterTextSplitter(
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
		)
		return splitter.split_text(text)

	def _get_bedrock_embeddings_client(self):
		"""Create a Bedrock embeddings client.

		We prefer `langchain_aws` but fall back to `langchain_community`.
		"""

		model_id = self._embedding_model_id
		if not model_id:
			raise ValueError("Missing required environment variable: BEDROCK_EMBEDDING_MODEL_ID")

		try:
			try:
				return LangchainAwsBedrockEmbeddings(model_id=model_id)
			except TypeError:
				return LangchainAwsBedrockEmbeddings(model=model_id) # type: ignore
		except Exception:
			pass

		try:
			try:
				return LangchainCommunityBedrockEmbeddings(model_id=model_id)
			except TypeError:
				return LangchainCommunityBedrockEmbeddings(model=model_id) # type: ignore
		except Exception as exc:
			raise DocumentServiceError(
				"LangChain Bedrock embeddings provider not available. Ensure langchain-aws or langchain-community is installed."
			) from exc

