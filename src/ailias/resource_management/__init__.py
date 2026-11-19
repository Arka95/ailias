"""One resource-lifecycle API for job finalization and dependency-safe retention."""

from ailias.resource_management.manager import ResourceManager, finalize_job_resources

__all__ = ["ResourceManager", "finalize_job_resources"]
