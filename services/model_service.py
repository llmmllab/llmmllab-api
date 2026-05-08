"""
ModelService — resolves the effective model for a request.

Rules:
1. If the caller specified a model and it is available on a runner, use it.
2. If the caller specified a model but it is *not* available, fall back to
   the user's ``default_model`` (from UserConfig).
3. If no ``default_model`` is configured, fall back to the first available
   TextToText model on any runner.
4. If the caller did *not* specify a model at all, use the user's
   ``default_model``, or the first available TextToText model.
5. If nothing is available, return the original (possibly unavailable)
   model so the downstream error path can handle it.
"""

from __future__ import annotations

from typing import Optional

from models import Model, ModelTask
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="model_service")


class ModelService:
    """Resolves model names against available runners.

    Singleton — use ``model_service`` from ``services``.
    """

    def __init__(self):
        self._cached_model_ids: Optional[set[str]] = None
        self._cached_models: Optional[dict[str, Model]] = None
        self._user_config_service = None

    def _get_user_config_service(self):
        if self._user_config_service is None:
            from services import user_config_service  # noqa: F811

            self._user_config_service = user_config_service
        return self._user_config_service

    async def resolve_default_model(
        self,
        requested_model: Optional[str],
        user_id: str,
    ) -> str:
        """Return the model ID that should be used for this request.

        Parameters
        ----------
        requested_model:
            The model name from the incoming API request (may be ``None``).
        user_id:
            The authenticated user ID (needed to look up ``default_model``).

        Returns
        -------
        str
            The resolved model ID.
        """
        # Fast path: requested model is available on a runner
        if requested_model:
            available = await self._available_model_ids()
            if requested_model in available:
                return requested_model

            # Requested model not found — try user's default_model
            fallback = await self._user_default_model(user_id)
            if fallback:
                logger.info(
                    "Requested model not available, falling back to default_model",
                    extra={
                        "user_id": user_id,
                        "requested": requested_model,
                        "fallback": fallback,
                    },
                )
                return fallback

            # No user default configured — try any available TextToText model
            any_available = await self._any_available_model()
            if any_available:
                logger.info(
                    "Requested model not available, using first available model",
                    extra={
                        "user_id": user_id,
                        "requested": requested_model,
                        "fallback": any_available,
                    },
                )
                return any_available

            # Nothing available — return original so downstream can error
            logger.warning(
                "Requested model not available and no default_model configured",
                extra={"user_id": user_id, "requested": requested_model},
            )
            return requested_model

        # No model specified — use user's default_model
        fallback = await self._user_default_model(user_id)
        if fallback:
            return fallback

        # No user default — try any available TextToText model
        any_available = await self._any_available_model()
        if any_available:
            logger.info(
                "No model specified, using first available model",
                extra={"user_id": user_id, "fallback": any_available},
            )
            return any_available

        # Nothing to fall back to — return empty so downstream errors
        logger.warning(
            "No model specified and no default_model configured",
            extra={"user_id": user_id},
        )
        return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _available_model_ids(self) -> set[str]:
        """Return the set of model IDs currently available on any runner."""
        if self._cached_model_ids is not None:
            return self._cached_model_ids

        models = await self._list_models()
        self._cached_model_ids = {m.id for m in models if m.id}
        return self._cached_model_ids

    async def _list_models(self) -> list[Model]:
        """Fetch the full model list from runners (cached)."""
        if self._cached_models is not None:
            return list(self._cached_models.values())

        from services.runner_client import runner_client

        try:
            models = await runner_client.list_models()
            self._cached_models = {m.id: m for m in models if m.id}
        except Exception as e:
            logger.warning(f"Failed to list models from runners: {e}")
            self._cached_models = {}

        return list(self._cached_models.values())

    async def get_model_by_id(self, model_id: str) -> Optional[Model]:
        """Look up a model by its ID from the runner cache.

        Returns the ``Model`` object if found, or ``None`` if the model
        is not available or the cache has not been populated yet.
        """
        if self._cached_models is None:
            await self._list_models()
        return self._cached_models.get(model_id) if self._cached_models else None

    async def _any_available_model(self) -> Optional[str]:
        """Return the ID of the first available TextToText model, or None."""
        try:
            from services.runner_client import runner_client

            model = await runner_client.model_by_task(ModelTask.TEXTTOTEXT)
            if model and model.id:
                return model.id
        except Exception as e:
            logger.warning(f"Failed to find any available model: {e}")
        return None

    async def _user_default_model(self, user_id: str) -> Optional[str]:
        """Look up the user's configured default_model."""
        try:
            config = await self._get_user_config_service().get_user_config(user_id)
            if config and hasattr(config, "default_model") and config.default_model:
                return config.default_model
        except Exception as e:
            logger.warning(
                f"Failed to load user config for default_model lookup: {e}"
            )

        return None

    def invalidate_cache(self) -> None:
        """Clear the cached model list (call when models change)."""
        self._cached_model_ids = None
        self._cached_models = None


# Singleton instance
model_service = ModelService()
