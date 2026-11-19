"""Inspect package dependencies without loading assets, materials or render resources."""

import json
from pathlib import Path

import unreal


def main(request, keep_alive=False):
    if request["operation"] != "inspect":
        raise ValueError(
            "Native asset deletion is deferred until the editor is stopped"
        )
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    registry.scan_paths_synchronous(["/Game/Ailias"], force_rescan=True)
    options = unreal.AssetRegistryDependencyOptions(
        include_hard_package_references=True,
        include_soft_package_references=True,
        include_searchable_names=True,
        include_hard_management_references=True,
        include_soft_management_references=True,
    )
    packages = sorted(
        {
            str(asset.package_name)
            for asset in registry.get_assets_by_path("/Game/Ailias", recursive=True)
        }
    )
    graph = {
        package: {
            "dependencies": sorted(
                str(value)
                for value in (registry.get_dependencies(package, options) or [])
            ),
            "referencers": sorted(
                str(value)
                for value in (registry.get_referencers(package, options) or [])
            ),
        }
        for package in packages
    }
    dirty = {
        str(package.get_name())
        for package in (
            list(unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages())
            + list(unreal.EditorLoadingAndSavingUtils.get_dirty_map_packages())
        )
    }
    result = {
        "status": "retention_completed",
        "graph": graph,
        "dirty_packages": sorted(dirty),
    }
    Path(request["receipt"]).write_text(json.dumps(result), encoding="utf-8")
