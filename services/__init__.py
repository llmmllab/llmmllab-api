"""
Service layer — business logic shared across all LLM interaction endpoints.

Services are stateless and composable.  Each service has a single clear
responsibility:

  • ``ToolService``           — server-tool separation and preparation
  • ``CompletionService``     — workflow execution, retry, continuation, nudge
  • ``UserConfigService``     — user configuration CRUD
  • ``ConversationService``   — conversation CRUD
  • ``MessageService``        — message CRUD
  • ``ApiKeyService``         — API key lifecycle and validation
  • ``TodoService``           — todo item CRUD
  • ``DocumentService``       — document upload and retrieval
  • ``MemoryService``         — memory search and storage
  • ``SummaryService``        — conversation summary retrieval
  • ``ModelService``          — resolve effective model (fallback to default_model)
"""

from .tool_service import ToolService, PreparedTools
from .completion_service import (
    CompletionService,
    CompletionResult,
    StreamAccumulator,
)
from .model_service import ModelService, model_service
from .user_config_service import UserConfigService, user_config_service
from .conversation_service import ConversationService, conversation_service
from .message_service import MessageService, message_service
from .api_key_service import ApiKeyService, api_key_service
from .todo_service import TodoService, todo_service
from .document_service import DocumentService, document_service
from .memory_service import MemoryService, memory_service
from .summary_service import SummaryService, summary_service

__all__ = [
    "ToolService",
    "PreparedTools",
    "CompletionService",
    "CompletionResult",
    "StreamAccumulator",
    "ModelService",
    "model_service",
    "UserConfigService",
    "user_config_service",
    "ConversationService",
    "conversation_service",
    "MessageService",
    "message_service",
    "ApiKeyService",
    "api_key_service",
    "TodoService",
    "todo_service",
    "DocumentService",
    "document_service",
    "MemoryService",
    "memory_service",
    "SummaryService",
    "summary_service",
]
