import unittest
from unittest.mock import Mock

import numpy as np

from ailias.imageprocessing.cuda import (
    CudaUnavailable, OpenCVCudaBackend, decode_yunet, graph_execution, require_cuda,
)


class CudaTests(unittest.TestCase):
    def runtime(self, devices=1, targets=None):
        cv = Mock()
        cv.cuda.getCudaEnabledDeviceCount.return_value = devices
        cv.dnn.DNN_TARGET_CUDA = 6
        cv.dnn.DNN_BACKEND_CUDA = 5
        cv.dnn.getAvailableTargets.return_value = [6] if targets is None else targets
        return cv

    def test_missing_cuda_device_fails_without_implicit_cpu_run(self):
        cv = self.runtime(devices=0)
        with self.assertRaises(CudaUnavailable):
            require_cuda(cv)
        cv.cuda.setDevice.assert_not_called()

    def test_cpu_only_dnn_is_refused_even_when_cuda_operators_are_available(self):
        cv = self.runtime(targets=[])
        with self.assertRaisesRegex(CudaUnavailable, "no CUDA DNN"):
            require_cuda(cv)
        cv.cuda.setDevice.assert_not_called()

    def test_explicit_cuda_request_selects_device_zero(self):
        cv = self.runtime()
        require_cuda(cv)
        cv.cuda.setDevice.assert_called_once_with(0)

    def test_cpu_fallback_is_refused_after_forward(self):
        with self.assertRaisesRegex(CudaUnavailable, "No CUDA layers"):
            graph_execution(r'layer [label="conv\nOCV/CPU"];')

    def test_graph_records_cpu_layers_honestly_alongside_cuda_layers(self):
        result = graph_execution(
            r'conv [label="convolution\nCUDA/CUDA"]; output [label="identity\nOCV/CPU"];'
        )
        self.assertEqual(result["cuda_graph_layers"], 1)
        self.assertEqual(result["graph_layers_by_backend_target"], {"CUDA/CUDA": 1, "OCV/CPU": 1})

    def test_yunet_decode_maps_grid_scores_boxes_and_landmarks(self):
        outputs = {}
        for stride in (8, 16, 32):
            count = (32 // stride) ** 2
            for name, channels in (("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10)):
                outputs[f"{name}_{stride}"] = np.zeros((1, count, channels), dtype=np.float32)
        outputs["cls_8"][0, 5] = 0.81
        outputs["obj_8"][0, 5] = 1
        outputs["bbox_8"][0, 5] = [0.5, 0.5, 0, 0]
        rows = decode_yunet(outputs, 32, 32, 0.85)
        self.assertEqual(len(rows), 1)
        np.testing.assert_allclose(rows[0][:4], [8, 8, 8, 8])
        np.testing.assert_allclose(rows[0][4:14], [8, 8] * 5)
        self.assertAlmostEqual(float(rows[0][14]), 0.9, places=6)

    def test_unused_model_is_reported_as_not_executed_not_falsely_verified(self):
        backend = object.__new__(OpenCVCudaBackend)
        backend.networks = {"yunet": Mock(), "sface": Mock()}
        backend.executions = {"yunet": {"status": "verified", "cuda_graph_layers": 8}}
        backend.provenance = {}
        report = backend.execution_report()
        self.assertEqual(report["sface"]["status"], "not_executed")
        self.assertEqual(report["yunet"]["status"], "verified")

    def test_gpu_resources_are_released_without_owning_any_global_runtime(self):
        backend = object.__new__(OpenCVCudaBackend)
        backend.networks = {"yunet": Mock()}
        backend.close()
        self.assertEqual(backend.networks, {})
