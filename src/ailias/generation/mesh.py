import math
from pathlib import Path


def ply_header(path: Path) -> dict:
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Native output is not a PLY file")
        result = {"elements": {}}
        for _ in range(256):
            line = stream.readline(4096)
            if not line:
                break
            fields = line.decode("ascii").split()
            if fields[:1] == ["format"]:
                result["format"] = fields[1]
            elif fields[:1] == ["element"]:
                count = int(fields[2])
                if count < 0:
                    raise ValueError("Negative PLY element count")
                result["elements"][fields[1]] = count
            elif fields == ["end_header"]:
                result["data_offset"] = stream.tell()
                return result
    raise ValueError("Missing or excessive PLY header")


def textured_ply_to_obj(source: Path, target: Path) -> dict:
    """Preserve COLMAP's per-corner UVs, which Blender's PLY importer discards."""
    header = ply_header(source)
    if header["format"] != "ascii":
        raise ValueError("Request COLMAP mesh_texturer --output_type TXT")
    vertices = header["elements"].get("vertex", 0)
    faces = header["elements"].get("face", 0)
    if min(vertices, faces) < 1:
        raise ValueError("Textured reconstruction has no surface")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream, target.open("w", encoding="ascii") as output:
        stream.seek(header["data_offset"])
        output.write(f"mtllib {target.stem}.mtl\nusemtl photographs\n")
        for _ in range(vertices):
            values = list(map(float, stream.readline().split()))
            if len(values) != 3 or not all(math.isfinite(v) for v in values):
                raise ValueError("Malformed textured PLY vertex")
            output.write("v " + " ".join(format(v, ".9g") for v in values) + "\n")
        for face_number in range(faces):
            fields = stream.readline().split()
            if len(fields) != 11 or fields[0] != b"3" or fields[4] != b"6":
                raise ValueError("Expected a triangle and six per-corner UV values")
            indices = list(map(int, fields[1:4]))
            uv = list(map(float, fields[5:]))
            if (not all(0 <= index < vertices for index in indices)
                    or not all(math.isfinite(v) for v in uv)):
                raise ValueError("Invalid textured face indices/coordinates")
            for index in range(0, 6, 2):
                output.write(f"vt {uv[index]:.9g} {uv[index + 1]:.9g}\n")
            output.write("f " + " ".join(
                f"{vertex + 1}/{3*face_number + corner + 1}"
                for corner, vertex in enumerate(indices)) + "\n")
    target.with_suffix(".mtl").write_text(
        "newmtl photographs\nKd 1 1 1\nmap_Kd texture.png\n", encoding="ascii")
    return {"vertices": vertices, "faces": faces, "texture": "photograph atlas, not measured albedo"}
