"""Read COLMAP cameras and restore source-image coordinates from portrait crops."""

import math
from pathlib import Path


def _records(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip() and not line.startswith("#"):
                yield line.split()


def _numbers(values):
    result = list(map(float, values))
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Non-finite recovered camera/point data")
    return result


def _rotation(quaternion):
    w, x, y, z = quaternion
    norm = math.sqrt(sum(v*v for v in quaternion))
    if abs(norm - 1) > 0.001:
        raise ValueError("Recovered camera quaternion is not normalized")
    return [
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ]


def read_model(directory: Path, inputs: dict) -> dict:
    """Reproject actual observations; retain SfM's unknown scale explicitly."""
    cameras = {}
    for fields in _records(directory / "cameras.txt"):
        model, width, height = fields[1], int(fields[2]), int(fields[3])
        values = _numbers(fields[4:])
        if model == "SIMPLE_PINHOLE" and len(values) == 3:
            fx, cx, cy = values
            fy = fx
        elif model == "PINHOLE" and len(values) == 4:
            fx, fy, cx, cy = values
        else:
            raise ValueError("Only undistorted pinhole camera models are supported")
        if min(width, height, fx, fy) <= 0:
            raise ValueError("Invalid recovered camera intrinsics")
        cameras[int(fields[0])] = (width, height, fx, fy, cx, cy)
    points = {}
    for fields in _records(directory / "points3D.txt"):
        xyz = _numbers(fields[1:4])
        if len(xyz) != 3:
            raise ValueError("Malformed recovered 3D point")
        points[int(fields[0])] = xyz
    entries = {entry["image_name"]: entry for entry in inputs["inputs"]["entries"]}
    recovered, all_errors, names = {}, [], set()
    with (directory / "images.txt").open(encoding="utf-8") as stream:
        while line := stream.readline():
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split(maxsplit=9)
            if len(fields) != 10 or fields[9].strip() not in entries:
                raise ValueError("Registered image is not in this frozen input set")
            name = fields[9].strip()
            if name in names:
                raise ValueError("Repeated registered image")
            names.add(name)
            entry = entries[name]
            pose = _numbers(fields[1:8])
            rotation = _rotation(pose[:4])
            translation = pose[4:]
            width, height, fx, fy, cx, cy = cameras[int(fields[8])]
            if entry["size"] != [width, height]:
                raise ValueError("Recovered image size differs from the native portrait")
            observations = stream.readline().split()
            if len(observations) % 3:
                raise ValueError("Malformed recovered 2D observations")
            errors = []
            for index in range(0, len(observations), 3):
                xy = _numbers(observations[index:index + 2])
                point_id = int(observations[index + 2])
                if point_id == -1:
                    continue
                if point_id not in points:
                    raise ValueError("Observation references an absent 3D point")
                point = points[point_id]
                projected = [sum(a*b for a, b in zip(row, point)) + offset
                             for row, offset in zip(rotation, translation)]
                if projected[2] <= 0:
                    raise ValueError("Triangulated observation is behind its camera")
                predicted = (fx*projected[0]/projected[2] + cx,
                             fy*projected[1]/projected[2] + cy)
                errors.append(math.dist(xy, predicted))
            if not errors:
                raise ValueError("Registered camera has no measured 3D observations")
            all_errors.extend(errors)
            x0, y0 = entry["portrait_bounds"][:2]
            recovered[entry["source_id"]] = {
                "image_name": name, "face_id": entry["face_id"],
                "world_to_camera": [row + [offset] for row, offset in zip(rotation, translation)]
                                   + [[0, 0, 0, 1]],
                "intrinsics": [[fx, 0, cx + x0], [0, fy, cy + y0], [0, 0, 1]],
                "portrait_intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                "portrait_size": [width, height],
                "reprojection_error_px": sum(errors)/len(errors),
                "observation_count": len(errors),
                "facial_landmarks_source": entry.get("landmarks_5_source", []),
                "view": entry.get("view", {"category": "unknown"}),
                "selection_weight": entry.get("selection_weight"),
            }
    return {
        "schema_version": 1, "convention": "OpenCV x-right, y-down, z-forward; world-to-camera",
        "world_units": "arbitrary_sfm; metric calibration required",
        "source_pixel_space": "EXIF-corrected full source; portrait offset restored in intrinsics",
        "cameras": recovered,
        "registered_images": len(recovered), "input_images": len(entries),
        "sparse_points": len(points),
        "mean_reprojection_error_px": sum(all_errors)/len(all_errors) if all_errors else None,
        "unregistered_images": sorted(set(entries) - names),
        "calibration_verified": False, "likeness_verified": False,
        "semantic_facial_constraints_used": False,
    }
