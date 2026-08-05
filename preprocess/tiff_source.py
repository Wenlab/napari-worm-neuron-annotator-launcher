"""Read TIFF frames and assemble source volumes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import tifffile


class TiffFrameSource:
    """Read numbered TIFF files or individual pages of a TIFF stack."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._stack: tifffile.TiffFile | None = None
        self._frame_paths: dict[int, Path] = {}
        self._directory_indexed = False

    def __enter__(self) -> Self:
        if self.path.is_dir():
            # Direct lookup avoids a full network-directory scan in the common
            # case where frames use predictable numeric names.
            pass
        elif self.path.is_file():
            self._stack = tifffile.TiffFile(self.path)
        else:
            raise FileNotFoundError(f"TIFF source does not exist: {self.path}")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._stack is not None:
            self._stack.close()

    def _index_directory(self) -> None:
        if self._directory_indexed:
            return
        for candidate in self.path.iterdir():
            if candidate.suffix.lower() not in {".tif", ".tiff"}:
                continue
            try:
                frame_number = int(candidate.stem)
            except ValueError:
                continue
            if frame_number in self._frame_paths:
                raise ValueError(
                    f"Duplicate TIFF frame number {frame_number} in {self.path}"
                )
            self._frame_paths[frame_number] = candidate
        self._directory_indexed = True
        if not self._frame_paths:
            raise ValueError(f"No numerically named TIFF frames found in {self.path}")

    def read(self, frame_number: int) -> np.ndarray:
        if self._stack is not None:
            if not 0 <= frame_number < len(self._stack.pages):
                raise IndexError(
                    f"TIFF page {frame_number} is outside "
                    f"[0, {len(self._stack.pages)}) for {self.path}"
                )
            frame = self._stack.pages[frame_number].asarray()
        else:
            direct_candidates = (
                self.path / f"{frame_number:08d}.tif",
                self.path / f"{frame_number:08d}.tiff",
                self.path / f"{frame_number}.tif",
                self.path / f"{frame_number}.tiff",
            )
            frame_path = next(
                (candidate for candidate in direct_candidates if candidate.is_file()),
                None,
            )
            if frame_path is None:
                self._index_directory()
                try:
                    frame_path = self._frame_paths[frame_number]
                except KeyError as error:
                    raise FileNotFoundError(
                        f"Missing TIFF frame {frame_number:08d} in {self.path}"
                    ) from error
            frame = tifffile.imread(frame_path)

        frame = np.asarray(frame)
        if frame.ndim != 2:
            raise ValueError(
                f"Expected a 2-D TIFF frame, got shape {frame.shape} at frame "
                f"{frame_number} in {self.path}"
            )
        return frame


def read_volume(
    source: TiffFrameSource,
    volume_number: int,
    frames_per_volume: int,
    z_start_frame: int,
    z_end_frame: int,
    reverse_z_by_volume_parity: Sequence[bool],
) -> np.ndarray:
    """Read one source volume as a ``(Z,Y,X)`` array."""

    frame_numbers = volume_frame_numbers(
        volume_number,
        frames_per_volume,
        z_start_frame,
        z_end_frame,
        reverse_z_by_volume_parity,
    )

    frames = [source.read(frame_number) for frame_number in frame_numbers]
    first_shape = frames[0].shape
    if any(frame.shape != first_shape for frame in frames):
        raise ValueError(
            f"Inconsistent TIFF shapes within source volume {volume_number}"
        )
    return np.stack(frames, axis=0)


def volume_frame_numbers(
    volume_number: int,
    frames_per_volume: int,
    z_start_frame: int,
    z_end_frame: int,
    reverse_z_by_volume_parity: Sequence[bool],
) -> list[int]:
    """Return source frame numbers in exported Z order."""

    frame_numbers = list(
        range(
            volume_number * frames_per_volume + z_start_frame,
            volume_number * frames_per_volume + z_end_frame + 1,
        )
    )
    if reverse_z_by_volume_parity[volume_number % 2]:
        frame_numbers.reverse()
    return frame_numbers
