"""Translate current photo evidence into explicit, uncertainty-labelled native appearance edits."""

import numpy as np

from ailias.facefitting.unreal.character_features_worker import validate_profile


GROOM_ROOT = "/MetaHumanCharacter/Optional/Grooms/Bindings"
HAIR_STYLES = (
    (.025, "Hair/WI_Hair_S_Clean"),
    (.05, "Hair/WI_Hair_M_Layered"),
    (float("inf"), "Hair/WI_Hair_L_Straight"),
)


def aggregate_observed_color(features, category, region=None):
    observed = []
    for item in features:
        value = item.get(category, {}) if region is None else item.get(category, {}).get(region, {})
        if value.get("status") == "observed":
            color = np.asarray(value["median_linear_rgb"], dtype=float)
            if color.shape == (3,) and np.isfinite(color).all() and np.all((color >= 0) & (color <= 1)):
                observed.append(color)
    if not observed:
        raise ValueError(f"No usable current-photo colour evidence for {region or category}")
    return np.median(observed, axis=0)


def build_groom_settings(asset, colour):
    # This is an appearance approximation, not an inverse measurement of hair pigment.
    luminance = float(np.asarray(colour) @ [.2126, .7152, .0722])
    return {"asset": GROOM_ROOT + "/" + asset,
            "parameters": {"Melanin": float(np.clip(1-np.sqrt(luminance), 0, 1))}}


def build_appearance_profile(evidence, baked, texture, sources, *, native_hair_asset=None):
    features = [face["features"] for face in evidence["faces"].values() if face["status"] == "observed"]
    if not features:
        raise ValueError("Native appearance requires freshly observed photo features")
    hair = aggregate_observed_color(features, "regions", "scalp_hair")
    brows = aggregate_observed_color(features, "regions", "eyebrows")
    iris = aggregate_observed_color(features, "iris")
    length = baked["hair_length_model_units"]
    if not isinstance(length, (int, float)) or not np.isfinite(length) or length < 0:
        raise ValueError("Native groom selection requires a finite estimated hair length")
    asset = next(asset for bound, asset in HAIR_STYLES if length < bound)
    hair_settings = build_groom_settings(asset, hair)
    if native_hair_asset is not None:
        hair_settings["asset"] = native_hair_asset
    profile = {
        "version": 1, "sources": sources, "face_basecolor": texture,
        "eyes": {"global_tint_linear": iris.tolist()},
        "grooms": {"Hair": hair_settings,
                   "Eyebrows": build_groom_settings("Eyebrows/WI_Eyebrows_M_Natural", brows)},
        "eyelashes": {"type": "short_fine", "enable_grooms": True,
                      "dye_color_linear": brows.tolist()},
    }
    estimates = {}
    for name, slot, directory, prefix in (
            ("beard", "Beard", "Beards", "Beard"),
            ("moustache", "Mustache", "Mustaches", "Mustache")):
        regions = [item.get("facial_hair_appearance", {}).get("regions", {}).get(name, {})
                   for item in features]
        regions = [item for item in regions if item.get("status") == "estimated"]
        if not regions:
            raise ValueError("Facial-hair region evidence is unresolved: " + name)
        fraction = float(np.median([item["dark_fraction"] for item in regions]))
        presence = sum(item["presence_estimate"] for item in regions)/len(regions) >= .5
        estimates[name] = {"presence_estimate": presence, "median_dark_fraction": fraction}
        if presence:
            colors = [item["median_dark_linear_rgb"] for item in regions
                      if item.get("median_dark_linear_rgb") is not None]
            color = np.median(colors, axis=0) if colors else hair
            size = "M" if fraction > .35 else "S"
            profile["grooms"][slot] = build_groom_settings(f"{directory}/WI_{prefix}_{size}_Stubble", color)
    return validate_profile(profile), {
        "source_features": len(features), "facial_hair_estimates": estimates,
        "native_hair_asset": hair_settings["asset"],
        "native_hair_selection": "configured" if native_hair_asset is not None else "length_estimate",
        "limitations": [
            "Observed photo colour retains lighting; iris tint is not an absolute calibrated iris colour",
            "Native stock groom style and melanin are approximations, not recovered individual strands",
            "Beard/moustache presence uses landmark-region darkness and can be confused by shadows",
            "Short-fine eyelashes are an explicit unmeasured native preset, not a parser measurement",
            "Photo texture retains skin, lip and visible facial-hair detail; unseen areas remain labelled fill",
        ],
    }
