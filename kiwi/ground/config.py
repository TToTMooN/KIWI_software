"""The ground stage's tunable parameters, in one frozen dataclass.

Constants that are not parameters live next to the code that uses them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GroundConfig:
    # windows
    window_s: float = 1.0
    """Window length [s]; windows start every window_s on the pose clock."""
    n_frames: int = 20
    """Frames per window; the frame stride follows the pose rate (6 at 120 fps)."""
    min_span_m: float = 0.10
    """Pose gate: horizontal camera travel inside the window."""
    max_off_nadir_deg: float = 55.0
    """Pose gate: median optical-axis angle from straight down."""
    window_stride: int = 1
    """Process every k-th gated window (1 = all); the others are flagged 'skipped'."""
    # views
    decode_size: int = 960
    """Frames are decoded at this square size (hardware resize on NVDEC)."""
    view_size: int = 518
    """Virtual view size [px]."""
    pitch_cap_deg: float = 25.0
    """View axis pitched clip(off_nadir - offset, 0, cap) deg toward the camera azimuth."""
    # identification
    cam_height_band: tuple[float, float] = (0.8, 1.8)
    """Plausible camera height above the floor [m]."""
    sweep_support_min: float = 0.05
    """Sweep: image fraction photo-consistent at a plane."""
    stereo_support_min: float = 0.01
    """Stereo: floor pixels as a fraction of scene-valid pixels."""
    agree_cm: float = 3.0
    """Sweep vs stereo agreement gate."""
    # sweep
    sweep_h_range: tuple[float, float] = (0.3, 1.9)
    """Camera-height hypotheses [m]."""
    sweep_dh_coarse: float = 0.02
    """Coarse step [m]."""
    sweep_dh_fine: float = 0.002
    # stereo
    stereo_b_range: tuple[float, float] = (0.15, 0.35)
    """Admissible baseline [m]."""
    stereo_pairs: int = 3
    """Pairs per window (median)."""
    stereo_tilt_deg: float = 5.0
    """FIXED plane tilt gate [deg]."""
    pose_sigma_m: float = 0.0034
    """Intra-window relative pose error in the per-window sigma."""
    # pooling / verdict
    outlier_mad: float = 3.0
    """Temporal gate: |z - running median| > min(outlier_mad x max(MAD, 1 cm), outlier_cap_cm)."""
    outlier_cap_cm: float = 5.0
    sigma_floor_m: float = 0.005
    """Pooled sigma never below this."""
    green_pair_sd_cm: float = 3.0
    green_gap_cm: float = 2.0
    green_used_frac: float = 0.4
    """Of the windows that PASS the pose gate."""
    min_used_per_cam: int = 20
    """Fewer accepted windows on a camera -> YELLOW."""
    device: str = "auto"
    """torch device; 'auto' = cuda when available."""

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["cam_height_band"] = list(d["cam_height_band"])
        d["sweep_h_range"] = list(d["sweep_h_range"])
        d["stereo_b_range"] = list(d["stereo_b_range"])
        return d
