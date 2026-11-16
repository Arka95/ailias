import sqlite3
from contextlib import closing
from pathlib import Path

COLMAP_PAIR_BASE = 2_147_483_647


def matching_summary(database: Path) -> dict:
    """Measure the verified view graph before spending time/VRAM on dense stereo."""
    try:
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            image_count = connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            pairs = connection.execute(
                "SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0").fetchall()
    except sqlite3.Error as error:
        raise ValueError("Native verified-match database is unreadable") from error
    neighbors = {}
    for pair, _ in pairs:
        first, second = divmod(pair, COLMAP_PAIR_BASE)
        neighbors.setdefault(first, set()).add(second)
        neighbors.setdefault(second, set()).add(first)
    remaining = set(neighbors)
    components = []
    while remaining:
        stack, size = [remaining.pop()], 0
        while stack:
            image = stack.pop()
            size += 1
            for other in neighbors[image]:
                if other in remaining:
                    remaining.remove(other)
                    stack.append(other)
        components.append(size)
    return {
        "input_images": image_count, "possible_pairs": image_count*(image_count - 1)//2,
        "verified_pairs": len(pairs), "images_with_verified_matches": len(neighbors),
        "largest_connected_component": max(components, default=0),
        "connected_component_sizes": sorted(components, reverse=True),
        "max_pair_inliers": max((inliers for _, inliers in pairs), default=0),
    }
