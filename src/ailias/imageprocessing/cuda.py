"""Run image-analysis models on the requested OpenCV CUDA device.

Checks actual DNN graph execution rather than assuming CUDA availability proves use.
"""

import re
from collections import Counter
from importlib.metadata import version

from PIL import Image

from ailias.geometry.metrics import align
from ailias.imageprocessing.backend import Face, detect_scaled_rows
from ailias.imageprocessing.models import REVISION, verified_models


class CudaUnavailable(ValueError):
    """Requested CUDA execution cannot be established; never retry on CPU."""


def require_cuda(cv):
    if cv.cuda.getCudaEnabledDeviceCount() < 1:
        raise CudaUnavailable("No CUDA device is available to this OpenCV build")
    if cv.dnn.DNN_TARGET_CUDA not in cv.dnn.getAvailableTargets(cv.dnn.DNN_BACKEND_CUDA):
        raise CudaUnavailable("OpenCV has no CUDA DNN target; use the existing GPU environment")
    cv.cuda.setDevice(0)


def graph_execution(graph):
    targets = Counter(re.findall(r"(CUDA|OCV)/([A-Z0-9_]+)", graph))
    cuda_layers = sum(count for (backend, _), count in targets.items() if backend == "CUDA")
    if not cuda_layers:
        raise CudaUnavailable("No CUDA layers observed after forward; refusing CPU fallback")
    return {"graph_layers_by_backend_target": {
        f"{backend}/{target}": count for (backend, target), count in targets.items()
    }, "cuda_graph_layers": cuda_layers, "status": "verified",
        "verification": "OpenCV graph backend/target dump after completed inference"}


def decode_yunet(outputs, width, height, threshold):
    import cv2
    import numpy as np

    candidates = []
    for stride in (8, 16, 32):
        scores = np.sqrt(np.clip(outputs[f"cls_{stride}"].reshape(-1), 0, 1)
                         * np.clip(outputs[f"obj_{stride}"].reshape(-1), 0, 1))
        indices = np.flatnonzero(scores >= threshold)
        if not len(indices):
            continue
        boxes = outputs[f"bbox_{stride}"].reshape(-1, 4)[indices]
        grid = np.column_stack((indices % (width // stride), indices // (width // stride)))
        centers = (grid + boxes[:, :2]) * stride
        sizes = np.exp(boxes[:, 2:]) * stride
        keypoints = outputs[f"kps_{stride}"].reshape(-1, 5, 2)[indices]
        keypoints = (keypoints + grid[:, None, :]) * stride
        candidates.extend(np.column_stack((
            centers - sizes / 2, sizes, keypoints.reshape(-1, 10), scores[indices],
        )).tolist())
    if not candidates:
        return []
    # Match the existing OpenCV adapter's integer-box NMS semantics.
    keep = cv2.dnn.NMSBoxes(
        [[int(value) for value in row[:4]] for row in candidates],
        [row[14] for row in candidates], threshold, 0.3, top_k=5000,
    )
    return [np.asarray(candidates[int(index)], dtype=np.float32) for index in np.asarray(keep).flatten()]


class OpenCVCudaBackend:
    def __init__(self, store, policy, body=True):
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise CudaUnavailable("Use an environment with CUDA-enabled OpenCV") from error
        paths = verified_models(store, body)
        self.cv, self.np, self.policy = cv2, np, policy
        self.networks = {}
        self.executions = {}
        cv2.setNumThreads(policy.cpu_threads)
        try:
            require_cuda(cv2)
            for name, path in paths.items():
                network = cv2.dnn.readNetFromONNX(str(path))
                network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.networks[name] = network
            self.provenance = {
                "provider": "opencv/opencv_zoo", "revision": REVISION,
                "backend": "opencv-cuda", "runtime": cv2.__version__,
                "opencv_module": cv2.__file__,
                "device": "cuda:0", "dtype": "float32",
                "cpu_threads": policy.cpu_threads, "opencv": cv2.__version__,
                "numpy": np.__version__, "pillow": version("Pillow"),
                "models": {name: str(path.relative_to(store.root)) for name, path in paths.items()},
                "execution": "CUDA model inference; CPU decoding/alignment/cropping/grouping",
                "cuda_device_count": cv2.cuda.getCudaEnabledDeviceCount(),
                "detection_scales": "configured side; no-face-only 960/640/320 retries, unchanged threshold",
            }
        except BaseException:
            self.close()
            raise

    def _run(self, name, tensor):
        network = self.networks[name]
        try:
            names = network.getUnconnectedOutLayersNames()
            network.setInput(tensor)
            outputs = network.forward(names)
            if name not in self.executions:
                self.executions[name] = graph_execution(network.dump())
            return dict(zip(names, outputs))
        except self.cv.error as error:
            raise CudaUnavailable(f"{name} CUDA inference failed; no CPU retry performed") from error

    def detect(self, image):
        cv, np = self.cv, self.np
        original = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
        def infer(small):
            height, width = small.shape[:2]
            padded = cv.copyMakeBorder(small, 0, (-height) % 32, 0, (-width) % 32,
                                      cv.BORDER_CONSTANT, value=0)
            predictions = self._run("yunet", cv.dnn.blobFromImage(padded))
            return decode_yunet(predictions, padded.shape[1], padded.shape[0], self.policy.detection_score)
        rows = detect_scaled_rows(cv, original, self.policy.detection_max_side, infer)
        target = np.asarray([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)
        faces = []
        for row in rows:
            landmarks = row[4:14].reshape(5, 2)
            _, transform = align(target, landmarks.astype(np.float64), fit="similarity")
            matrix = np.column_stack((
                transform["scale"] * np.asarray(transform["rotation"]).T,
                transform["translation"],
            ))
            aligned = cv.warpAffine(original, matrix, (112, 112), flags=cv.INTER_LINEAR)
            blob = cv.dnn.blobFromImage(aligned, swapRB=True)
            feature = next(iter(self._run("sface", blob).values())).flatten()
            feature /= max(float(np.linalg.norm(feature)), 1e-12)
            faces.append(Face(row[:4].tolist(), landmarks.tolist(), float(row[14]), feature.tolist()))
        return sorted(faces, key=lambda face: (face.box[0], face.box[1]))

    def segment(self, image):
        if "pphumanseg" not in self.networks:
            raise ValueError("Body segmentation is disabled")
        cv, np = self.cv, self.np
        reduced = image.copy()
        reduced.thumbnail((self.policy.detection_max_side, self.policy.detection_max_side))
        blob = cv.dnn.blobFromImage(np.asarray(reduced), 1 / 127.5, (192, 192), (127.5,) * 3)
        scores = next(iter(self._run("pphumanseg", blob).values()))[0].transpose(1, 2, 0)
        scores = cv.resize(scores, reduced.size, interpolation=cv.INTER_LINEAR)
        mask = Image.fromarray(np.argmax(scores, axis=2).astype(np.uint8) * 255)
        return mask.resize(image.size, Image.Resampling.NEAREST)

    def execution_report(self):
        reports = {
            name: self.executions.get(name, {"status": "not_executed"})
            for name in self.networks
        }
        self.provenance["verified_execution"] = reports
        return reports

    def close(self):
        self.networks.clear()
