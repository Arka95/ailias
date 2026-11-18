"""Fail native acceptance on the shader fallback that produced checkerboard skin."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ailias.generation.tools import NativeTools, ToolError

from ailias.generation.unreal_validation import validate_shader_output, validate_unreal_log


class UnrealShaderValidationTests(unittest.TestCase):
    def test_active_platform_must_be_sm6_not_merely_a_d3d12_capable_device(self):
        validate_shader_output('LogCsvProfiler: shaderplatform="PCD3D_SM6"', require_platform=True)
        for text in ('shaderplatform="PCD3D_SM5"', "D3D12 GPU supports shader model 6.7", ""):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "active PCD3D_SM6"):
                validate_shader_output(text, require_platform=True)

    def test_material_fallback_is_failure_even_when_native_bake_export_says_success(self):
        for error in (
                "LogShaderCompilers: Warning: Failed to compile Material /Plugin/M_Skin, Default Material will be used",
                "error X4510: maximum ps_5_0 sampler register index (16) exceeded"):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, "checkerboard"):
                validate_shader_output('shaderplatform="PCD3D_SM6"\n' + error
                                       + "\nLogData: Exporting finished\nfeatures_saved")

    def test_unrelated_native_warnings_do_not_fail_a_correct_sm6_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "native.log"
            path.write_text('shaderplatform="PCD3D_SM6"\n'
                            "LogWindows: Failed to load aqProf.dll\n"
                            "LogHairStrands: Warning: matching material fallback\n"
                            "Exporting finished\n")
            validate_unreal_log(path)

    def test_standalone_function_thumbnails_do_not_fail_valid_runtime_materials(self):
        previews = "\n".join(
            "[AssetLog] /Game/Runtime/Common/Lookdev_UHM/Skin/Material_Functions/"
            f"{name}.{name}:Material_0: Failed to compile Material for platform PCD3D_SM6\n"
            "Missing Preview connection for function input 'Enabled'"
            for name in ("MF_skin_fuzzApply", "MF_skin_bakedInputs"))
        validate_shader_output(previews)
        with self.assertRaisesRegex(ValueError, "checkerboard"):
            validate_shader_output(previews + "\nFailed to compile Material /Game/M_Skin_Baked")

    def test_zero_exit_one_shot_cannot_publish_shader_fallback(self):
        from types import SimpleNamespace

        def native(*args, stdout, **kwargs):
            stdout.write(b'shaderplatform="PCD3D_SM6"\n'
                         b"Failed to compile Material /Plugin/M_Skin\nExporting finished\n")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temporary, \
                patch("ailias.generation.tools.subprocess.run", side_effect=native):
            tools = NativeTools(Path(temporary))
            tools.paths["unreal"] = Path(temporary) / "UnrealEditor-Cmd.exe"
            with self.assertRaisesRegex(ToolError, "refusing fallback/checkerboard"):
                tools.run("unreal", "cold-check", ["-d3d12", "-sm6"], Path(temporary))
