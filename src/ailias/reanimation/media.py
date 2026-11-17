"""Validate bounded PCM narration and encode exact contiguous portrait frames with explicit GPU policy."""

import re
import subprocess
import wave
from pathlib import Path

from PIL import Image


def inspect_audio(path, maximum=120):
    with wave.open(str(path), "rb") as stream:
        frames, rate = stream.getnframes(), stream.getframerate()
        if (stream.getcomptype() != "NONE" or stream.getsampwidth() != 2 or
                stream.getnchannels() != 1 or not 16000 <= rate <= 96000 or frames <= 0):
            raise ValueError("Narration must be nonempty mono PCM16 WAV, 16-96 kHz")
        duration = frames/rate
        if duration > maximum:
            raise ValueError("Narration exceeds the configured reel duration")
        peak = 0
        actual_frames = 0
        import array
        while raw := stream.readframes(65536):
            samples = array.array("h", raw)
            actual_frames += len(samples)
            peak = max(peak, max(map(abs, samples), default=0))
        if actual_frames != frames:
            raise ValueError("Narration WAV is truncated")
        if peak == 0:
            raise ValueError("Silent narration cannot produce a talking reel")
    return {"duration_seconds": duration, "sample_rate": rate, "samples": frames,
            "channels": 1, "sample_width": 2, "peak": peak}


def inspect_frames(directory, policy, count):
    paths = sorted(directory.glob("frame.*.png"))
    expected = [f"frame.{index:06d}.png" for index in range(count)]
    if [path.name for path in paths] != expected:
        raise ValueError("Render must contain the exact contiguous expected frame range starting at zero")
    for path in paths:
        with Image.open(path) as image:
            if image.size != (policy.width, policy.height):
                raise ValueError("Rendered frame dimensions do not match the portrait policy")
            image.verify()
    return paths


def encode_command(executable, frames, audio, output, policy, count):
    if count <= 0:
        raise ValueError("Video requires positive frame count")
    return [str(executable), "-hide_banner", "-nostdin", "-n", "-framerate", str(policy.fps),
            "-start_number", "0", "-i", str(frames / "frame.%06d.png"),
            "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-frames:v", str(count),
            "-c:v", policy.encoder, "-pix_fmt", "yuv420p",
            * (["-preset", "p5", "-cq", "19"] if policy.encoder == "h264_nvenc"
               else ["-preset", "medium", "-crf", "19"]),
            "-c:a", "aac", "-b:a", "192k", "-af", "apad", "-t",
            f"{count/policy.fps:.9f}", "-movflags", "+faststart", str(output)]


def verify_video(executable, video, policy, expected_count):
    metadata = subprocess.run([str(executable), "-hide_banner", "-i", str(video)],
                              capture_output=True, text=True, timeout=30, check=False)
    details = metadata.stderr
    if (not re.search(rf"\b{policy.width}x{policy.height}\b", details) or
            "Video: h264" not in details or "Audio: aac" not in details or
            not re.search(rf"\b{policy.fps}(?:\.0+)? fps\b", details)):
        raise ValueError("Encoded video must contain H.264 portrait video and AAC audio")
    command = [str(executable), "-hide_banner", "-nostdin", "-i", str(video),
               "-map", "0:v:0", "-an", "-fps_mode", "passthrough",
               "-f", "null", "-progress", "pipe:1", "-"]
    decoded = subprocess.run(command, capture_output=True, text=True,
                             timeout=policy.timeout_seconds, check=False)
    counts = re.findall(r"^frame=(\d+)$", decoded.stdout, re.MULTILINE)
    if decoded.returncode or not counts or int(counts[-1]) != expected_count:
        raise ValueError("Final MP4 decoded frame count differs from the requested range")
    expected_duration = expected_count/policy.fps
    def decoded_duration(process):
        times = re.findall(r"^out_time_us=(\d+)$", process.stdout, re.MULTILINE)
        if not times:
            raise ValueError("Decoded stream did not report its actual duration")
        return int(times[-1])/1_000_000
    video_duration = decoded_duration(decoded)
    if abs(video_duration-expected_duration) > 1/policy.fps + .001:
        raise ValueError("Decoded video timing differs from the requested duration")
    audio_decode = subprocess.run([str(executable), "-v", "error", "-nostdin", "-i", str(video),
                                   "-map", "0:a:0", "-f", "null", "-progress", "pipe:1", "-"],
                                  capture_output=True, text=True, timeout=policy.timeout_seconds, check=False)
    if audio_decode.returncode:
        raise ValueError("Final MP4 audio stream is not decodable")
    audio_duration = decoded_duration(audio_decode)
    if audio_duration < expected_duration-.02 or audio_duration > expected_duration+.12:
        raise ValueError("Decoded audio does not cover the requested reel duration")
    return {"decoded_frames": expected_count, "dimensions": [policy.width, policy.height],
            "duration_seconds": expected_duration, "decoded_video_seconds": video_duration,
            "decoded_audio_seconds": audio_duration, "audio_decoded": True,
            "encoder_requested": policy.encoder}
