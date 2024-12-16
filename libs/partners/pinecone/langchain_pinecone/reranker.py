from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Union

from pinecone import Pinecone
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import ConfigDict

class PineconeRerank(BaseDocumentCompressor):
    """Document compressor that uses Pinecone's Rerank API."""

    client: Pinecone = None
    """Pinecone client to use for reranking documents."""
    model: str = "pinecone-rerank-v0"
    """Model to use for reranking."""
    top_n: Optional[int] = 5
    """Number of documents to return."""
    truncation: str = "END"
    """Truncation strategy for reranking."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.client = Pinecone()

    def _rerank(
        self,
        documents: Sequence[Union[str, Document]],
        query: str,
    ) -> Any:
        """Returns an ordered list of documents ordered by their relevance
        to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
        """
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        return self.client.inference.rerank(
            model=self.model,
            query=query,
            documents=docs,
            top_n=self.top_n,
            return_documents=True,
            parameters={"truncate": self.truncation},
        )

    async def _arerank(
        self,
        documents: Sequence[Union[str, Document]],
        query: str,
    ) -> Any:
        """Returns an ordered list of documents ordered by their relevance
        to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
        """
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        return await self.client.inference.rerank(
            model=self.model,
            query=query,
            documents=docs,
            top_n=self.top_n,
            return_documents=True,
            parameters={"truncate": self.truncation},
        )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Pinecone's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents in relevance_score order.
        """
        if len(documents) == 0:
            return []

        compressed = []
        results = self._rerank(documents, query)
        for res in results.data:
            doc = documents[res.index]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = round(res.score, 4)
            compressed.append(doc_copy)
        return compressed

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Pinecone's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents in relevance_score order.
        """
        if len(documents) == 0:
            return []

        compressed = []
        results = await self._arerank(documents, query)
        for res in results.data:
            doc = documents[res.index]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = round(res.score, 4)
            compressed.append(doc_copy)
        return compressed
