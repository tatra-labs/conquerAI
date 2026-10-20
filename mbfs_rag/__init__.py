"""MBFS customer-knowledge retrieval and grounding service.

from mbfs_rag import KnowledgeService
service = KnowledgeService.load()
answer = service.answer("What's the fee for an outgoing international wire on Plus?")
"""

from .service import Answer, KnowledgeService

__all__ = ["Answer", "KnowledgeService"]
