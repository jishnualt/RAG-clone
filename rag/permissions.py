from rest_framework.permissions import BasePermission
from rest_framework import exceptions


ALLOWED_NOT_READY_VIEWS = {
    "register",
    "login",
    "logout",
    "refresh",
    "llm-setup",
    "me-status",
    "user-status",
    "llm-config-create",
    "llm-config-list",
    "llm-config-detail",
    "openai-key-create",
    "openai-key-list",
    "openai-key-detail",
}


class LLMReadyPermission(BasePermission):
    """
    Blocks access to protected APIs until a user has configured an LLM and collection.
    """

    message = {
        "code": "LLM_SETUP_REQUIRED",
        "message": "Configure an LLM and create a collection to use this API.",
    }

    def has_permission(self, request, view):
        view_name = getattr(request.resolver_match, "view_name", None)
        if view_name in ALLOWED_NOT_READY_VIEWS:
            return True

        user = getattr(request, "user", None)
        ready = bool(
            getattr(user, "llm_configured", False)
            and getattr(user, "active_collection_ready", False)
            and getattr(user, "active_collection", None)
        )
        if ready:
            return True

        detail = dict(self.message)
        detail["llm_configured"] = bool(getattr(user, "llm_configured", False))
        detail["active_collection_ready"] = bool(getattr(user, "active_collection_ready", False))
        raise exceptions.PermissionDenied(detail=detail)
