import hashlib
import io
import os
import tempfile
from pathlib import Path

from PIL import Image

from ailias.storage import Store, digest


def encode_png(image: Image.Image) -> bytes:
    clean = Image.new(image.mode, image.size)
    clean.paste(image)
    buffer = io.BytesIO()
    clean.save(buffer, format="PNG")
    return buffer.getvalue()


def store_png(store: Store, image: Image.Image, role: str, scratch: Path,
              created: list[str]) -> str:
    return store_png_bytes(store, encode_png(image), role, scratch, created)


def store_png_bytes(store: Store, content: bytes, role: str, scratch: Path,
                    created: list[str]) -> str:
    if role not in {"portraits", "review", "bodies", "masks", "cleansed", "cutouts", "alphas"}:
        raise ValueError("Unsupported generated image role")
    checksum = hashlib.sha256(content).hexdigest()
    relative = str(Path("imageprocessing") / "assets" / role / f"{checksum}.png")
    destination = store.resolve(relative)
    if destination.exists():
        if digest(destination) != checksum:
            raise ValueError("Existing generated asset failed its content hash")
        return relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=scratch, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
        os.replace(temporary, destination)
        created.append(relative)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return relative
