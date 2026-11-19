"""Compatibility imports; resource lifecycle implementation is in resource_management."""

from ailias.resource_management.job_resources import (
    JOB_COLLECTIONS, artifact_paths, cleanup_owned_files, finalize_resources, is_link,
    job_directory, metadata_documents, owned_files, reference_path, referenced_paths,
    remove_empty_directories,
)

__all__ = [
    "JOB_COLLECTIONS", "artifact_paths", "cleanup_owned_files", "finalize_resources",
    "is_link", "job_directory", "metadata_documents", "owned_files", "reference_path",
    "referenced_paths", "remove_empty_directories",
]
