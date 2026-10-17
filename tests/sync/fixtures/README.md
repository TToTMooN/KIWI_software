# Real-audio sync fixtures

Three 12-second mono audio excerpts, used by `test_real_pair.py` (the same-model
X5 pair) and `test_real_mixed_pair.py` (each X5 against the GO 3, exercising the
shipped A/V table) to run the sync recipe against actually-recorded audio and
check it against an **optical** ground truth.

## Provenance

Cut from a rolling-QR sync capture. In that capture both cameras filmed
one monitor showing a rolling QR clock; fitting each camera to that clock
independently, then differencing, pins the true cross-camera offset. The
monitor's display latency and refresh phase are common to both cameras and
cancel in the difference.

| | |
|---|---|
| x5a | `VID_20260725_183802_00_057.insv` (55.68 s), serial IAHYA2511CWJEA = `insta360-X5-001` |
| x5b | `VID_20260725_223755_00_012.insv` (60.07 s), serial IAHYA25116W5XY = `insta360-X5-003` |
| go3 | `VID_20260725_183800_00_006.mp4` (64.66 s), Insta360 GO 3, stored 1440x2560 @ 29.97 with a -90 display matrix |
| camera (x5 pair) | 2x Insta360 X5, both 1920x1920 @ 119.88, AAC 48 kHz |
| **optical ground truth (x5a->x5b)** | **-4.3604964 s**, standard error **0.57 ms** (`kiwi calib qr sync`, `delta_qr_ms`) |
| **optical ground truth (x5a->go3)** | **-5.8436 s**, SE **2.4 ms** (OCR text-clock fallback) |
| **optical ground truth (x5b->go3)** | **-1.4831 s**, SE **2.4 ms** (OCR text-clock fallback) |
| estimator, whole clips (x5a->x5b) | **-4.3624 s** |

The GO 3's optical truth comes from OCR of the clock's text line (Theil-Sen
fit, SE 2.4 ms).

Sign convention matches the estimator's throughout: positive means the second
clip started later.

## How they were cut

Decoded with the recipe pinned in `kiwi.sync.audio.EXTRACTION`:

```
ffmpeg -map 0:a:0 -ac 1 -ar 48000 -f s16le -
```

then, from the decoded samples:

```
x5a = A[36.0 s : 48.0 s]
x5b = B[36.0 s + 161395 samples : 48.0 s + 161395 samples]
go3 = G[36.0 s + 233717 samples : 48.0 s + 233717 samples]
```

then low-passed and stored as int16 in `np.savez_compressed`.

**The second cut is shifted** by the whole-clip offset, so the excerpt keeps
the shared content and stays anchored to the whole-clip value: the expected
offset is exactly

```
-4.3624 + 161395/48000 = -1.000004 s
```

so `test_real_pair.py` asserts that number, *and* asserts that undoing the shift
returns both the whole-clip -4.3624 and the optical -4.3604964.

The go3 cut follows the same rule against x5a: 233717 samples is the whole-clip
audio offset (-5.8691 s) plus ~1.0 s, so its expected excerpt offset is
-0.999996 s against x5a and +0.000308 s against x5b (shift difference
233717 - 161395 = 72322 samples), with 11 s and 12 s of shared content
respectively.

**The mixed pair is refine-only.** `test_real_mixed_pair.py` seeds
`refine_offset` with the documented cut geometry and pins the refinement plus
the registry-resolved A/V correction against the optical truth.

## The low-pass

The excerpts are band-limited so intelligible speech is impossible by
construction:

```
scipy.signal.butter(6, 150.0, "lowpass", fs=48000, output="sos")
scipy.signal.sosfiltfilt(sos, x)          # zero-phase, identical on both clips
```

then a single common gain brings the x5 pair's peak to 0.70 full scale (the
`audio_non_silent` and `audio_unclipped` gates read absolute sample values, so
the level has to land in range). The go3 excerpt is scaled to 0.70 on its own.

The lowest vowel formant, F1 of a close vowel, sits near 250 Hz; a 150 Hz
cut removes every formant. `sosfiltfilt` is zero-phase and the same filter is
applied to every clip, so it introduces no group delay and the lag between
them is unchanged.

`test_fixture_is_band_limited` and `test_go3_fixture_is_band_limited` enforce
the low-pass, and `test_the_pair_still_defeats_a_raw_argmax` checks that a raw
GCC-PHAT argmax still false-locks on the x5 pair, so a regenerated fixture
cannot quietly drop either property.

## Regenerating

Regenerating requires the original recordings; if the fixtures are ever rebuilt,
update the expected values in `test_real_pair.py` and `test_real_mixed_pair.py`
in the same commit.
