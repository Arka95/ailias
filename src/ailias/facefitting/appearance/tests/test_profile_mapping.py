"""Current-run evidence reaches native texture, iris and groom consumers without old profiles."""

import json
from pathlib import Path
import tempfile
import unittest

from ailias.facefitting.appearance.profile_mapping import build_appearance_profile
from ailias.facefitting.configuration import DEFAULT_APPEARANCE_CONFIG, load_appearance_policy


class NativePhotoProfileTests(unittest.TestCase):
    def fixture(self):
        color = {"status": "observed", "median_linear_rgb": [.1, .05, .025]}
        region = {"status": "estimated", "dark_fraction": .4, "presence_estimate": True,
                  "median_dark_linear_rgb": [.02, .015, .01]}
        features = {"regions": {"scalp_hair": color, "eyebrows": color}, "iris": color,
                    "facial_hair_appearance": {"regions": {"beard": region, "moustache": region}}}
        return {"faces": {"frame": {"status": "observed", "features": features}}}

    def test_current_colour_texture_iris_and_facial_hair_have_explicit_native_consumers(self):
        source = {"path": "fresh.json", "sha256": "a"*64}
        profile, mapping = build_appearance_profile(self.fixture(), {"hair_length_model_units": .02},
                                         "/Game/Current/PhotoBaseColor", [source])
        self.assertEqual(profile["face_basecolor"], "/Game/Current/PhotoBaseColor")
        self.assertEqual(profile["eyes"]["global_tint_linear"], [.1, .05, .025])
        self.assertEqual(set(profile["grooms"]), {"Hair", "Eyebrows", "Beard", "Mustache"})
        self.assertTrue(profile["grooms"]["Hair"]["asset"].endswith("WI_Hair_S_Clean"))
        self.assertEqual(profile["sources"], [source])
        self.assertTrue(mapping["facial_hair_estimates"]["beard"]["presence_estimate"])
        self.assertIn("shadows", " ".join(mapping["limitations"]))

    def test_unobserved_iris_cannot_silently_substitute_a_personal_preset(self):
        evidence = self.fixture()
        evidence["faces"]["frame"]["features"]["iris"] = {"status": "review"}
        with self.assertRaisesRegex(ValueError, "colour evidence for iris"):
            build_appearance_profile(evidence, {"hair_length_model_units": .02}, "/Game/Texture", [])

    def test_explicit_native_hairstyle_overrides_the_length_heuristic(self):
        hair = "/MetaHumanCharacter/Optional/Grooms/Bindings/Hair/WI_Hair_S_SweptUp"
        profile, mapping = build_appearance_profile(
            self.fixture(), {"hair_length_model_units": .1}, "/Game/Texture", [],
            native_hair_asset=hair, assembly={"pipeline": "cinematic"})
        self.assertEqual(profile["grooms"]["Hair"]["asset"], hair)
        self.assertEqual(mapping["native_hair_selection"], "configured")
        self.assertIn("Melanin", profile["grooms"]["Hair"]["parameters"])
        self.assertEqual(profile["assembly"], {"pipeline": "cinematic"})

    def test_missing_facial_hair_inference_blocks_instead_of_skipping_it(self):
        evidence = self.fixture()
        del evidence["faces"]["frame"]["features"]["facial_hair_appearance"]
        with self.assertRaisesRegex(ValueError, "Facial-hair region evidence is unresolved"):
            build_appearance_profile(evidence, {"hair_length_model_units": .02}, "/Game/Texture", [])

    def test_native_complexion_uses_controls_instead_of_projected_photo_texture(self):
        settings = {"skin": {"u": .35, "v": .9, "roughness": 1.12, "face_texture_index": 7},
                    "accents": {"lips": {"lightness": .46}}}
        hair = "/MetaHumanCharacter/Grooms/SelectedHair"
        profile, mapping = build_appearance_profile(
            self.fixture(), {}, None, [], native_hair_asset=hair, native_appearance=settings)
        self.assertEqual(profile["skin"], settings["skin"])
        self.assertEqual(profile["accents"], settings["accents"])
        self.assertIsNone(profile["face_basecolor"])
        self.assertEqual(profile["grooms"]["Hair"]["asset"], hair)
        self.assertEqual(mapping["skin_source"], "native_controls")
        with self.assertRaisesRegex(ValueError, "Native appearance"):
            build_appearance_profile(self.fixture(), {}, None, [], native_hair_asset=hair,
                                     native_appearance={"face_basecolor": "/Game/Bad"})

    def test_assembly_policy_is_validated_without_requiring_native_appearance_edits(self):
        policy = json.loads(DEFAULT_APPEARANCE_CONFIG.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "appearance.json"
            policy["assembly"] = {"pipeline": "cinematic"}
            path.write_text(json.dumps(policy))
            self.assertEqual(load_appearance_policy(path)["assembly"], policy["assembly"])
            policy["assembly"]["pipeline"] = "unknown"
            path.write_text(json.dumps(policy))
            with self.assertRaisesRegex(ValueError, "Assembly pipeline"):
                load_appearance_policy(path)

    def test_native_appearance_policy_validates_controls_and_requires_explicit_hairstyle(self):
        policy = json.loads(DEFAULT_APPEARANCE_CONFIG.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "appearance.json"
            policy["native_appearance"] = {"skin": {"u": .35, "v": .9}}
            path.write_text(json.dumps(policy))
            with self.assertRaisesRegex(ValueError, "native_hair_asset"):
                load_appearance_policy(path)
            policy["native_hair_asset"] = "/MetaHumanCharacter/Grooms/SelectedHair"
            path.write_text(json.dumps(policy))
            self.assertEqual(load_appearance_policy(path)["native_appearance"], policy["native_appearance"])
            for settings in ({"skin": {"v": 2}}, {"sculpt": {}}, {}, []):
                policy["native_appearance"] = settings
                path.write_text(json.dumps(policy))
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    load_appearance_policy(path)

    def test_absence_evidence_does_not_force_beard_or_moustache(self):
        evidence = self.fixture()
        regions = evidence["faces"]["frame"]["features"]["facial_hair_appearance"]["regions"]
        for region in regions.values():
            region.update(presence_estimate=False, dark_fraction=.01)
        profile, _ = build_appearance_profile(evidence, {"hair_length_model_units": .02}, "/Game/Texture", [])
        self.assertEqual(set(profile["grooms"]), {"Hair", "Eyebrows"})
