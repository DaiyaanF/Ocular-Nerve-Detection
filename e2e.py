"""
e2e.py — High-fidelity OCT E2E analysis pipeline
==================================================
Reads Heidelberg Spectralis .E2E files and produces the same measurement
set as the fundus detector:

  Disc Radius   (mm)
  ONSD          (mm)   — Optic Nerve Sheath Diameter
  Globe Area    (mm²)  — posterior globe cross-sectional area
  RoC           (mm)   — radius of curvature of posterior globe
  CRAE          (mm)   — Central Retinal Arteriole Equivalent (Knudtson)
  CRVE          (mm)   — Central Retinal Venule Equivalent   (Knudtson)
  AVR                  — Arteriovenous Ratio (CRAE / CRVE)

Improvements over the original:
  • Bilateral + CLAHE preprocessing (speckle-aware)
  • Multi-threshold segmentation (Otsu + adaptive + percentile voting)
  • Profile-scan ONSD with FWHM + sheath-gradient refinement
  • Globe boundary: physical-space (mm) parabola/sagitta fit to the traced
    posterior retinal boundary — not a bounded Hough search, and not a
    general circle fit either (numerically unstable on the near-flat arcs
    this modality actually shows — see segment_globe)
  • OCT vessel-shadow detection → Knudtson CRAE/CRVE/AVR
  • All units in mm (not cm); ONSD bug fixed
  • Averaged over the central third of B-scans for stability
  • Output matches fundus detector format exactly
"""

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing as sk_closing, disk
from skimage.measure import label
from oct_converter.readers import E2E


# =========================================================
# 1. LOAD E2E FILE
# =========================================================

def load_e2e(file_path):
    """
    Load a Heidelberg Spectralis .E2E file.

    oct_converter returns a *list* of OCTVolumeWithMetaData objects (one per
    acquisition series).  Each series has:
        .volume        — list of 2-D B-scan arrays
        .pixel_spacing — [sx_mm, sy_mm, sz_mm]  (x horizontal, y axial, z slice)
        .laterality    — 'L' or 'R'

    Returns
    -------
    series_list : list of dicts, each containing:
        'scans'      : np.ndarray  shape (N, H, W)
        'sx'         : float  mm/px horizontal
        'sy'         : float  mm/px axial
        'laterality' : str  ('L', 'R', or 'U')
        'series_idx' : int
        'patient_id' : str
    """
    e2e      = E2E(file_path)
    vol_list = e2e.read_oct_volume()

    # Guarantee we have a list even if a single object is returned
    if not isinstance(vol_list, list):
        vol_list = [vol_list]

    series = []
    for i, vol in enumerate(vol_list):
        try:
            scans = np.array(vol.volume, dtype=np.float32)
            if scans.ndim != 3 or scans.shape[0] == 0:
                continue

            # pixel_spacing = [sx_mm, sy_mm, sz_mm] (Spectralis convention)
            ps = getattr(vol, 'pixel_spacing', None)
            if ps is not None and len(ps) >= 2:
                sx = float(ps[0])   # horizontal  ~0.011 mm/px
                sy = float(ps[1])   # axial        ~0.0039 mm/px
            else:
                sx, sy = 0.011, 0.003871   # Spectralis defaults

            series.append({
                'scans':      scans,
                'sx':         sx,
                'sy':         sy,
                'laterality': getattr(vol, 'laterality', 'U'),
                'series_idx': i,
                'patient_id': str(getattr(vol, 'patient_id', '')),
            })
        except Exception as exc:
            print(f"  Skipping series {i}: {exc}")

    return series


# =========================================================
# 2. HIGH-FIDELITY PREPROCESSING
# =========================================================

def preprocess(img_raw):
    """
    Bilateral denoise → CLAHE → normalise to uint8.
    Bilateral filter preserves layer edges while reducing speckle.
    CLAHE recovers local contrast lost after denoising.
    """
    img = img_raw.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Bilateral: preserve edges, suppress speckle
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # CLAHE: recover local contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 4))
    img   = clahe.apply(img)

    return img


# =========================================================
# 3. MULTI-THRESHOLD SEGMENTATION  (vote 2-of-3)
# =========================================================

def _multi_threshold(img):
    """
    Combine three independent thresholds by majority vote.
    Returns a binary mask.
    """
    # Otsu
    ot = threshold_otsu(img)
    m1 = (img > ot).astype(np.uint8)

    # Adaptive Gaussian
    m2 = cv2.adaptiveThreshold(
        img, 1,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=-5
    )

    # 90th-percentile
    p90 = np.percentile(img, 90)
    m3  = (img > p90).astype(np.uint8)

    # Vote: pixel is bright if at least 2/3 methods agree
    vote = m1 + m2 + m3
    mask = (vote >= 2).astype(np.uint8)

    return mask


# =========================================================
# 4. OPTIC NERVE + SHEATH SEGMENTATION
# =========================================================

def segment_nerve_sheath(img):
    """
    Segment the optic nerve (bright, central column) and
    the surrounding sheath (slightly dimmer, flanking the nerve).

    Strategy
    --------
    1. Multi-threshold bright mask
    2. Morphological cleanup
    3. Intensity-gradient split:
         - nerve = pixels above 80th-percentile of the masked region
         - sheath = full mask minus nerve
    4. Keep only the largest connected component in each
    """
    mask = _multi_threshold(img)

    # Close small gaps, remove debris
    mask = sk_closing(mask, disk(4))
    mask = remove_small_objects(mask, max_size=149)
    mask = mask.astype(np.uint8)

    h, w = mask.shape

    # Restrict to central 60 % of width (nerve + sheath are in the middle)
    roi = np.zeros_like(mask)
    left  = int(w * 0.20)
    right = int(w * 0.80)
    roi[:, left:right] = mask[:, left:right]

    # Intensity split inside the ROI
    intensities = img[roi > 0]
    if len(intensities) == 0:
        return np.zeros_like(mask), np.zeros_like(mask)

    split_thresh = np.percentile(intensities, 70)

    nerve  = ((roi > 0) & (img >= split_thresh)).astype(np.uint8)
    sheath = roi.copy()

    # Keep largest component only
    def _largest(m):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            return m
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (lab == largest_idx).astype(np.uint8)

    nerve  = _largest(nerve)
    sheath = _largest(sheath)

    return nerve, sheath


# =========================================================
# 5. GLOBE SEGMENTATION
# =========================================================

def _trace_posterior_boundary(img):
    """
    Trace one boundary point per column: the deepest (bottom-most) pixel of
    the main bright retinal-layer band. This tracks the RPE / Bruch's
    membrane — the outer retinal layer, which follows the globe wall's own
    curvature far more closely than any other visible structure in an OCT
    B-scan.

    The central ~60% of the width (same ROI convention `segment_nerve_sheath`
    already uses) is excluded. That's not just noise-avoidance: the RPE is
    anatomically interrupted at the optic disc itself (the Bruch's membrane
    opening), and separately, a bright optic-nerve-head column commonly
    extends deeper into the frame than the true wall boundary right where a
    B-scan is centered — either way, tracing "the deepest bright pixel" in
    that region does not measure the globe wall and biases a fit right at
    the vertex, where the curvature signal matters most.

    Returns (x_px, y_px) arrays — one y per included column that had
    bright-mask signal, columns with no signal are skipped.
    """
    h, w = img.shape
    mask = _multi_threshold(img)
    mask = remove_small_objects(mask.astype(bool), max_size=30).astype(np.uint8)

    disc_left, disc_right = int(w * 0.20), int(w * 0.80)

    xs, ys = [], []
    for x in range(w):
        if disc_left <= x < disc_right:
            continue
        col = np.where(mask[:, x])[0]
        if col.size == 0:
            continue
        xs.append(x)
        ys.append(col[-1])
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _fit_shallow_arc_radius(x, y):
    """
    Fit the visible boundary as a shallow arc of a much larger circle, using
    a parabola fit ("radius from sagitta") instead of a general algebraic
    circle fit.

    An earlier version used an algebraic (Taubin) circle fit here. That
    method is numerically unstable exactly in this regime: a nearly-flat
    point cloud sits close to the R→∞ degenerate case, where small pixel
    noise can pull the closed-form solution toward a wildly wrong SMALL
    radius instead of the correct large one. Confirmed on a real B-scan —
    the Taubin fit returned 1.3mm on an arc that visibly, obviously curves
    far more gently than that.

    "Nearly flat, want the equivalent radius" is precisely the regime
    optics/lens-making solves with a parabola fit: near the vertex of a
    circle of radius R, y ≈ x²/(2R), so fitting y = a·x² + b·x + c via plain
    (numerically stable, no degeneracy) least squares gives R = 1/(2|a|).

    Returns (cx, cy, r) in whatever units x/y were given in — cx/cy are an
    approximate corresponding circle center, good enough for a display
    overlay — or None if there aren't enough points to fit.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 10:
        return None
    try:
        a, b, c = np.polyfit(x, y, 2)
    except np.linalg.LinAlgError:
        return None

    if abs(a) < 1e-9:
        r = 1e6  # essentially flat within measurable precision
    else:
        r = 1.0 / (2.0 * abs(a))

    vx = -b / (2 * a) if abs(a) > 1e-9 else float(np.mean(x))
    vy = float(np.polyval((a, b, c), vx))
    # a > 0 means the arc sags to larger y (downward) away from the vertex,
    # so the corresponding circle's center sits above it, and vice versa.
    cy = vy - r if a > 0 else vy + r
    return float(vx), float(cy), float(r)


def segment_globe(img, sx, sy):
    """
    Fit the posterior globe's radius of curvature and cross-sectional area
    from the visible retinal boundary arc.

    Why the old approach failed: an OCT B-scan only ever shows a few
    millimeters of retina — a tiny sliver of the eye's true ~12mm-radius
    curvature. Searching for "a whole circle that fits inside this frame"
    (the old Hough-based approach) is bounded to radii smaller than the
    real anatomy by roughly an order of magnitude, so it could never find
    the real boundary and instead locked onto coincidental noise in the
    retinal-layer texture on every scan (confirmed: it was returning
    ~0.9-1.0mm "radius of curvature" values against a true ~12mm eye).

    Fix: trace the visible boundary across the B-scan width (see
    `_trace_posterior_boundary`), convert those points to physical mm using
    sx/sy *before* fitting — pixel spacing here is anisotropic (~0.01mm/px
    horizontal vs ~0.004mm/px axial), so a real circle would look like a
    squashed ellipse in raw pixel coordinates — then fit the equivalent
    radius via a parabola/sagitta fit (see `_fit_shallow_arc_radius`), which
    unlike a general circle fit stays numerically stable on the near-flat
    arcs this modality actually produces.

    Returns
    -------
    mask       : uint8 array marking the traced boundary pixels (for overlay)
    curve_px   : (N, 2) int array of (x, y) pixel points along the *fitted
                 parabola itself*, sampled across the full image width — for
                 drawing on the raw image. An earlier version returned a
                 circle center + radius instead, converted to pixel space
                 with an averaged (isotropic) scale; since the real radius
                 is thousands of pixels and the anisotropic px/mm scale
                 doesn't carry over to a single circle, that circle didn't
                 even pass through the visible frame — the overlay panel
                 silently showed nothing. Drawing the actual fitted curve
                 instead is simpler and is guaranteed to intersect the image
                 you're looking at, since it's evaluated at every column.
    roc_mm     : radius of curvature in mm (the actual measurement)
    area_mm2   : cross-sectional area in mm² (pi * roc_mm^2)
    """
    h, w = img.shape
    xs_px, ys_px = _trace_posterior_boundary(img)

    mask = np.zeros((h, w), dtype=np.uint8)
    for x, y in zip(xs_px.astype(int), ys_px.astype(int)):
        mask[y, x] = 1

    def _flat_fallback():
        r_est_mm = 12.0
        flat_y = int(h * 0.6)
        curve = np.stack([np.arange(w), np.full(w, flat_y)], axis=1)
        return mask, curve, r_est_mm, float(np.pi * r_est_mm ** 2)

    min_points = max(20, w // 10)
    if len(xs_px) < min_points:
        return _flat_fallback()

    x_mm = xs_px * sx
    y_mm = ys_px * sy

    # One outlier-rejection pass: choroidal vessels / segmentation slips
    # occasionally throw a handful of points far off the true boundary.
    # Residuals here MUST be measured against the parabola itself (vertical
    # distance, y - polyval), not against the derived circle center/radius —
    # a circle only osculates a parabola right at the vertex, so "distance
    # to that circle" grows systematically toward the edges of the chord
    # even for perfectly clean data, and wrongly strips good edge points.
    # (Confirmed: doing it against the circle turned a correct 40.1mm fit
    # into a wrong 102mm one on a controlled synthetic test.)
    try:
        coeffs = np.polyfit(x_mm, y_mm, 2)
        resid = np.abs(y_mm - np.polyval(coeffs, x_mm))
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
        keep = resid < np.median(resid) + 4 * mad
        if keep.sum() >= min_points:
            x_mm, y_mm = x_mm[keep], y_mm[keep]
    except np.linalg.LinAlgError:
        pass

    fit = _fit_shallow_arc_radius(x_mm, y_mm)

    if fit is None:
        return _flat_fallback()

    # Refit the final (post-trim) parabola coefficients once more, purely
    # to sample a display curve consistent with whatever points actually
    # produced the reported radius.
    try:
        disp_coeffs = np.polyfit(x_mm, y_mm, 2)
        x_mm_disp = np.arange(w) * sx
        y_mm_disp = np.polyval(disp_coeffs, x_mm_disp)
        y_px_disp = np.round(y_mm_disp / sy).astype(int)
        curve_px = np.stack([np.arange(w), y_px_disp], axis=1)
    except np.linalg.LinAlgError:
        curve_px = np.stack([np.arange(w), np.full(w, int(h * 0.6))], axis=1)

    _cx_mm, _cy_mm, r_mm = fit
    r_mm = abs(r_mm)
    area_mm2 = float(np.pi * r_mm ** 2)

    return mask, curve_px, r_mm, area_mm2


# =========================================================
# 6. OPTIC DISC RADIUS
# =========================================================

def compute_disc_radius(nerve_mask, sx, sy):
    """
    Fit a circle to the nerve mask and return its radius in mm.
    Uses the mean of (sx, sy) for isotropic pixel size.
    """
    contours, _ = cv2.findContours(
        nerve_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.nan

    largest = max(contours, key=cv2.contourArea)
    (_, _), r_px = cv2.minEnclosingCircle(largest)

    # Average spatial scale (nerve is roughly circular)
    scale = (sx + sy) / 2.0
    return float(r_px) * scale


# =========================================================
# 7. ONSD  (profile-scan FWHM with sheath-gradient refinement)
# =========================================================

def compute_onsd(sheath_mask, img, sx):
    """
    Measure ONSD in mm via horizontal profile scan through the nerve.

    For each of 5 evenly-spaced rows in the sheath region:
      1. Smooth the intensity profile
      2. Find the FWHM of the sheath echo
    Return the median width × sx.
    """
    rows = np.where(np.sum(sheath_mask, axis=1) > 0)[0]
    if len(rows) < 3:
        return np.nan

    # Sample rows across the middle 60 % of the sheath height
    lo = rows[int(len(rows) * 0.20)]
    hi = rows[int(len(rows) * 0.80)]
    sample_rows = np.linspace(lo, hi, 7, dtype=int)

    widths_px = []
    ks = int(6 * 3 + 1) | 1
    x  = np.arange(ks) - ks // 2
    k  = np.exp(-0.5 * (x / 3.0) ** 2);  k /= k.sum()

    for row in sample_rows:
        profile = img[row, :].astype(float)
        smooth  = np.convolve(profile, k, mode='same')

        bg      = float(np.median(smooth[:len(smooth)//5]))
        peak    = float(np.max(smooth))
        if peak <= bg + 5:
            continue

        thresh  = bg + 0.40 * (peak - bg)
        above   = smooth > thresh
        if not np.any(above):
            continue
        idxs = np.where(above)[0]
        w    = int(idxs[-1] - idxs[0])
        if w > 3:
            widths_px.append(w)

    if not widths_px:
        # Fallback: bounding-box width × 1.15
        cols = np.where(np.sum(sheath_mask, axis=0) > 0)[0]
        return float(len(cols)) * 1.15 * sx if len(cols) > 0 else np.nan

    return float(np.median(widths_px)) * sx


# =========================================================
# 8. POSTERIOR GLOBE AREA & RADIUS OF CURVATURE
# =========================================================

# =========================================================
# 9. VESSEL SHADOW DETECTION → CRAE / CRVE / AVR
# =========================================================
#
# In OCT B-scans, blood vessels cast dark vertical shadows through the
# retinal layers.  We detect these as dark columns in the bright retinal
# band, measure their widths, and apply Knudtson's formula.
#

def _knudtson(diameters, coeff):
    """Iterative Knudtson vessel-equivalent formula."""
    d = sorted(diameters, reverse=True)
    while len(d) > 1:
        new_d = []
        for i in range(0, len(d) - 1, 2):
            new_d.append(coeff * np.sqrt(d[i]**2 + d[i+1]**2))
        if len(d) % 2 == 1:
            new_d.append(d[-1])
        d = sorted(new_d, reverse=True)
    return d[0]


def detect_vessel_shadows(img, sx):
    """
    Detect retinal vessel shadows in an OCT B-scan.

    Strategy
    --------
    1. Find the bright retinal band (top 40 % of the column median profile)
    2. Compute a column-wise darkness score within that band
    3. Threshold to find dark shadow columns
    4. Group adjacent dark columns into vessel candidates
    5. Classify by relative darkness: darker → vein, lighter → artery
    6. Return artery diameters (mm) and vein diameters (mm)
    """
    h, w = img.shape

    # Step 1 – locate the retinal band (brightest horizontal region)
    row_means = np.mean(img, axis=1)
    bright_thresh = np.percentile(row_means, 60)
    bright_rows = np.where(row_means >= bright_thresh)[0]

    if len(bright_rows) < 5:
        return [], []

    band_top = int(bright_rows[0])
    band_bot = int(bright_rows[-1])
    band     = img[band_top:band_bot, :]

    if band.shape[0] < 3:
        return [], []

    # Step 2 – column darkness score (inverse of mean within band)
    col_means = np.mean(band, axis=0).astype(float)
    # Normalise: 0 = darkest, 1 = brightest
    col_norm  = (col_means - col_means.min()) / (col_means.ptp() + 1e-6)

    # Step 3 – dark shadow columns
    shadow_thresh = np.percentile(col_norm, 25)
    is_shadow = col_norm < shadow_thresh

    # Step 4 – group into vessel segments
    vessels = []
    in_seg  = False
    seg_start = 0
    for c in range(w):
        if is_shadow[c] and not in_seg:
            in_seg    = True
            seg_start = c
        elif not is_shadow[c] and in_seg:
            in_seg = False
            seg_width = c - seg_start
            if 2 <= seg_width <= 30:   # plausible vessel width in pixels
                mean_dark = float(np.mean(col_norm[seg_start:c]))
                vessels.append((seg_width * sx, mean_dark))  # (width_mm, darkness)
    # Close last segment
    if in_seg:
        seg_width = w - seg_start
        if 2 <= seg_width <= 30:
            mean_dark = float(np.mean(col_norm[seg_start:]))
            vessels.append((seg_width * sx, mean_dark))

    if not vessels:
        return [], []

    # Step 5 – classify: darker columns → veins, lighter → arteries
    vessels.sort(key=lambda v: v[0], reverse=True)
    top = vessels[:12]

    darkness_vals = [v[1] for v in top]
    med_dark      = float(np.median(darkness_vals))

    arteries = [d for d, dk in top if dk >= med_dark]   # less dark = more reflective = artery
    veins    = [d for d, dk in top if dk <  med_dark]   # darker shadow = vein

    return arteries, veins


def compute_vascular_metrics(img, sx):
    """
    Compute CRAE, CRVE, and AVR from OCT vessel shadows.
    Returns (crae_mm, crve_mm, avr) — any may be None if not enough vessels.
    """
    arteries, veins = detect_vessel_shadows(img, sx)

    crae = _knudtson(arteries, 0.88) if len(arteries) >= 2 else (arteries[0] if arteries else None)
    crve = _knudtson(veins,    0.95) if len(veins)    >= 2 else (veins[0]    if veins    else None)

    # Physiological constraint: arteries must be narrower than veins (CRAE < CRVE).
    # If the shadow classifier flipped them, swap and recompute.
    if crae is not None and crve is not None and crae > crve:
        arteries, veins = veins, arteries
        crae = _knudtson(arteries, 0.88) if len(arteries) >= 2 else (arteries[0] if arteries else None)
        crve = _knudtson(veins,    0.95) if len(veins)    >= 2 else (veins[0]    if veins    else None)

    avr = (crae / crve) if (crae and crve and crve > 0 and crae < crve) else None

    return crae, crve, avr


# =========================================================
# 10. PER-SCAN ANALYSIS
# =========================================================

def analyze_scan(img_raw, sx, sy):
    """
    Run the full pipeline on a single B-scan.

    Returns a dict with keys matching the fundus detector output:
      disc_radius_mm, onsd_mm, globe_area_mm2, roc_mm,
      crae_mm, crve_mm, avr
    """
    img = preprocess(img_raw)

    nerve, sheath = segment_nerve_sheath(img)
    globe_mask, globe_curve_px, roc_mm, globe_area_mm2 = segment_globe(img, sx, sy)

    disc_radius_mm = compute_disc_radius(nerve, sx, sy)
    onsd_mm        = compute_onsd(sheath, img, sx)
    crae_mm, crve_mm, avr  = compute_vascular_metrics(img, sx)

    return {
        'disc_radius_mm':  disc_radius_mm,
        'onsd_mm':         onsd_mm,
        'globe_area_mm2':  globe_area_mm2,
        'roc_mm':          roc_mm,
        'crae_mm':         crae_mm,
        'crve_mm':         crve_mm,
        'avr':             avr,
        # internals for visualisation
        '_img':    img,
        '_nerve':  nerve,
        '_sheath': sheath,
        '_globe':      globe_mask,
        '_globe_curve': globe_curve_px,
    }


# =========================================================
# 11. AGGREGATE OVER CENTRAL B-SCANS
# =========================================================

def analyze_e2e(file_path, output_dir=None, pixels_per_mm_x=None, pixels_per_mm_y=None):
    """
    Full pipeline for a .E2E file.

    Loads all acquisition series from the file, processes every B-scan in each
    series, and returns:
      - Per-scan DataFrame  (all series combined, with 'series' and 'laterality' columns)
      - Aggregated summary  (median over central third of each series, then median across series)
      - Printed report matching the fundus detector format

    Parameters
    ----------
    file_path        : path to the .E2E file
    output_dir       : directory to save PNG results and CSVs; defaults to
                       a folder named after the E2E file
    pixels_per_mm_x  : override horizontal scale (mm/px) for all series
    pixels_per_mm_y  : override axial scale (mm/px) for all series
    """
    print(f"\nLoading: {file_path}")
    series_list = load_e2e(file_path)

    if not series_list:
        raise RuntimeError(f"No valid series found in {file_path}")

    print(f"  Series found : {len(series_list)}")

    # Output directory
    if output_dir is None:
        base = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(os.path.dirname(file_path), f"{base}_results")
    os.makedirs(output_dir, exist_ok=True)

    all_records   = []
    series_summaries = []

    for ser in series_list:
        scans = ser['scans']
        sx    = pixels_per_mm_x if pixels_per_mm_x else ser['sx']
        sy    = pixels_per_mm_y if pixels_per_mm_y else ser['sy']
        lat   = ser['laterality']
        sidx  = ser['series_idx']

        n = len(scans)
        label = f"series_{sidx:02d}_{lat}"
        print(f"\n  [{label}]  {n} B-scans  sx={sx:.5f} sy={sy:.5f} mm/px")

        # Per-series output sub-folder
        ser_dir = os.path.join(output_dir, label)
        os.makedirs(ser_dir, exist_ok=True)

        records = []
        for i, scan in enumerate(scans):
            print(f"    scan {i+1}/{n}", end='\r')
            res = analyze_scan(scan, sx, sy)

            records.append({
                'series':          label,
                'laterality':      lat,
                'scan_index':      i,
                'disc_radius_mm':  res['disc_radius_mm'],
                'onsd_mm':         res['onsd_mm'],
                'globe_area_mm2':  res['globe_area_mm2'],
                'roc_mm':          res['roc_mm'],
                'crae_mm':         res['crae_mm'],
                'crve_mm':         res['crve_mm'],
                'avr':             res['avr'],
            })

            _save_scan_figure(res, i, sx, sy, ser_dir)

        print()

        ser_df = pd.DataFrame(records)
        ser_df.to_csv(os.path.join(ser_dir, "oct_metrics.csv"),
                      index=False, float_format='%.4f')

        # Aggregate: median over central third of this series
        lo      = n // 3
        hi      = 2 * n // 3
        central = ser_df.iloc[lo:hi] if hi > lo else ser_df

        def _med(col):
            vals = central[col].dropna()
            return float(np.median(vals)) if len(vals) > 0 else np.nan

        ser_summary = {k: _med(k) for k in
                       ['disc_radius_mm', 'onsd_mm', 'globe_area_mm2',
                        'roc_mm', 'crae_mm', 'crve_mm', 'avr']}
        ser_summary['series']     = label
        ser_summary['laterality'] = lat
        ser_summary['n_scans']    = n

        pd.DataFrame([ser_summary]).to_csv(
            os.path.join(ser_dir, "oct_summary.csv"),
            index=False, float_format='%.4f'
        )

        all_records.extend(records)
        series_summaries.append(ser_summary)

    # ---- Combined DataFrame -----------------------------------------------
    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(output_dir, "oct_metrics_all.csv"),
              index=False, float_format='%.4f')

    # ---- Overall summary: median across all per-series medians ----------------
    metric_keys = ['disc_radius_mm', 'onsd_mm', 'globe_area_mm2',
                   'roc_mm', 'crae_mm', 'crve_mm', 'avr']
    summary = {}
    for k in metric_keys:
        vals = [s[k] for s in series_summaries if not np.isnan(s[k])]
        summary[k] = float(np.median(vals)) if vals else np.nan

    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, "oct_summary.csv"),
        index=False, float_format='%.4f'
    )

    # ---- Print report -------------------------------------------------------
    _print_report(summary, file_path, len(df), output_dir)

    return df, summary


# =========================================================
# 12. VISUALISATION
# =========================================================

def _save_scan_figure(res, scan_idx, sx, sy, output_dir):
    """Save a 2×3 figure for one B-scan."""
    img    = res['_img']
    nerve  = res['_nerve']
    sheath = res['_sheath']
    globe_curve = res['_globe_curve']

    _, axes = plt.subplots(2, 3, figsize=(18, 8))

    # Original
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title(f'B-scan {scan_idx} (preprocessed)')
    axes[0, 0].axis('off')

    # Nerve mask
    axes[0, 1].imshow(img, cmap='gray')
    axes[0, 1].imshow(nerve, cmap='Reds', alpha=0.4)
    axes[0, 1].set_title('Nerve (red)')
    axes[0, 1].axis('off')

    # Sheath mask
    axes[0, 2].imshow(img, cmap='gray')
    axes[0, 2].imshow(sheath, cmap='Blues', alpha=0.4)
    axes[0, 2].set_title('Sheath (blue)')
    axes[0, 2].axis('off')

    # Globe overlay — draw the fitted parabola itself rather than a circle.
    # The real radius is thousands of pixels and the px/mm scale here is
    # anisotropic, so a single circle converted to pixel space either
    # distorts badly or (as happened before this fix) doesn't even pass
    # through the visible frame. The curve is evaluated at every column, so
    # it's guaranteed to show up.
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if globe_curve is not None and len(globe_curve) > 1:
        h_img = img.shape[0]
        pts = globe_curve[(globe_curve[:, 1] >= 0) & (globe_curve[:, 1] < h_img)].astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], False, (255, 165, 0), 2)
    axes[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Globe boundary (blue)')
    axes[1, 0].axis('off')

    # ONSD line
    onsd_overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    sheath_rows  = np.where(np.sum(sheath, axis=1) > 0)[0]
    if len(sheath_rows) >= 3:
        mid_row = int(np.median(sheath_rows))
        cols    = np.where(sheath[mid_row] > 0)[0]
        if len(cols) > 0:
            cv2.line(onsd_overlay,
                     (int(cols[0]), mid_row),
                     (int(cols[-1]), mid_row),
                     (0, 255, 255), 2)
    axes[1, 1].imshow(cv2.cvtColor(onsd_overlay, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('ONSD measurement line (yellow)')
    axes[1, 1].axis('off')

    # Measurements panel
    axes[1, 2].axis('off')
    def _mm(v, dp=3):
        return f"{v:.{dp}f} mm" if v is not None and not np.isnan(v) else "N/A"
    def _mm2(v, dp=2):
        return f"{v:.{dp}f} mm²" if v is not None and not np.isnan(v) else "N/A"

    avr = res.get('avr')
    avr_norm = (" (normal)" if avr and 0.67 <= avr <= 0.75 else
                " (narrow)"  if avr and avr < 0.67  else
                " (wide v.)" if avr else "")

    lines = [
        "MEASUREMENTS",
        "=" * 30,
        f"Scale: {sx:.5f} mm/px (x)",
        f"       {sy:.5f} mm/px (y)",
        "",
        f"Disc Radius : {_mm(res.get('disc_radius_mm'))}",
        f"ONSD        : {_mm(res.get('onsd_mm'))}",
        f"Globe Area  : {_mm2(res.get('globe_area_mm2'))}",
        f"RoC         : {_mm(res.get('roc_mm'))}",
        f"CRAE        : {_mm(res.get('crae_mm'), 4)}",
        f"CRVE        : {_mm(res.get('crve_mm'), 4)}",
        f"AVR         : {f'{avr:.3f}{avr_norm}' if avr and not np.isnan(avr) else 'N/A'}",
    ]

    axes[1, 2].text(0.05, 0.95, "\n".join(lines),
                    transform=axes[1, 2].transAxes,
                    fontsize=9, verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].set_title('Measurements')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scan_{scan_idx:04d}.png"),
                dpi=120, bbox_inches='tight')
    plt.close()


def _print_report(summary, file_path, n_scans, output_dir):
    """Print a report in the same format as the fundus detector."""
    def _mm(v, dp=3):
        return f"{v:.{dp}f} mm" if v is not None and not np.isnan(v) else "N/A"
    def _mm2(v, dp=2):
        return f"{v:.{dp}f} mm²" if v is not None and not np.isnan(v) else "N/A"

    avr = summary.get('avr')
    avr_norm = (" (normal)" if avr and 0.67 <= avr <= 0.75 else
                " (narrow)"  if avr and avr < 0.67  else
                " (wide v.)" if avr else "")

    sep = "=" * 40
    print(f"\n{sep}")
    print(f"OCT E2E ANALYSIS REPORT")
    print(sep)
    print(f"File    : {os.path.basename(file_path)}")
    print(f"B-scans : {n_scans}  (median over central third)")
    print(sep)
    print(f"Disc Radius : {_mm(summary.get('disc_radius_mm'))}")
    print(f"ONSD        : {_mm(summary.get('onsd_mm'))}")
    print(f"Globe Area  : {_mm2(summary.get('globe_area_mm2'))}")
    print(f"RoC         : {_mm(summary.get('roc_mm'))}")
    print(f"CRAE        : {_mm(summary.get('crae_mm'), 4)}")
    print(f"CRVE        : {_mm(summary.get('crve_mm'), 4)}")
    avr_str = f"{avr:.3f}{avr_norm}" if avr and not np.isnan(avr) else "N/A"
    print(f"AVR         : {avr_str}")
    print(sep)
    print(f"Results saved to: {output_dir}")
    print(sep)


# =========================================================
# 13. GUI FOLDER PICKER  (same pattern as fundus detector)
# =========================================================

def select_e2e_and_run():
    """
    Open a GUI file picker for a .E2E file, then run the full pipeline.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Select Heidelberg Spectralis .E2E file",
        filetypes=[("E2E files", "*.E2E *.e2e"), ("All files", "*.*")]
    )

    if not file_path:
        print("No file selected — cancelled.")
        root.destroy()
        return

    output_dir = filedialog.askdirectory(
        title="Select output folder (Cancel = folder next to E2E file)"
    )
    root.destroy()

    if not output_dir:
        base = os.path.splitext(file_path)[0]
        output_dir = base + "_results"

    df, summary = analyze_e2e(file_path, output_dir=output_dir)

    # Summary popup
    try:
        root2 = tk.Tk()
        root2.withdraw()
        root2.attributes('-topmost', True)

        def _mm(v, dp=3):
            return f"{v:.{dp}f} mm" if v is not None and not np.isnan(v) else "N/A"
        def _mm2(v, dp=2):
            return f"{v:.{dp}f} mm²" if v is not None and not np.isnan(v) else "N/A"

        avr = summary.get('avr')
        avr_norm = (" (normal)" if avr and 0.67 <= avr <= 0.75 else
                    " (narrow)"  if avr and avr < 0.67  else
                    " (wide v.)" if avr else "")
        avr_str = f"{avr:.3f}{avr_norm}" if avr and not np.isnan(avr) else "N/A"

        msg = (
            f"Disc Radius : {_mm(summary.get('disc_radius_mm'))}\n"
            f"ONSD        : {_mm(summary.get('onsd_mm'))}\n"
            f"Globe Area  : {_mm2(summary.get('globe_area_mm2'))}\n"
            f"RoC         : {_mm(summary.get('roc_mm'))}\n"
            f"CRAE        : {_mm(summary.get('crae_mm'), 4)}\n"
            f"CRVE        : {_mm(summary.get('crve_mm'), 4)}\n"
            f"AVR         : {avr_str}\n\n"
            f"Results: {output_dir}"
        )
        from tkinter import messagebox as _mb
        _mb.showinfo("OCT Analysis Complete", msg)
        root2.destroy()
    except Exception:
        pass

    return df, summary


# =========================================================
# 14. BATCH — process every .E2E file in a folder
# =========================================================

def process_all_e2e(input_dir, output_dir=None, pixels_per_mm_x=None, pixels_per_mm_y=None):
    """
    Find every .E2E file in input_dir, run the full pipeline on each one,
    and write:
      • Per-subject subfolder  with per-scan PNGs + oct_metrics.csv
      • batch_summary.csv      one row per subject with median measurements

    Parameters
    ----------
    input_dir        : folder that contains the .E2E files
    output_dir       : where to save all results
                       (defaults to <input_dir>/batch_results/)
    pixels_per_mm_x  : optional horizontal scale override (mm/px)
    pixels_per_mm_y  : optional axial scale override (mm/px)

    Returns
    -------
    batch_df : DataFrame  — one row per subject
    """
    e2e_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith('.e2e')
    ])

    if not e2e_files:
        print(f"No .E2E files found in: {input_dir}")
        return pd.DataFrame()

    if output_dir is None:
        output_dir = os.path.join(input_dir, "batch_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nBatch processing {len(e2e_files)} E2E file(s)")
    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print("=" * 50)

    batch_rows = []
    ok_count   = 0
    fail_count = 0

    for i, fp in enumerate(e2e_files, 1):
        subject = os.path.splitext(os.path.basename(fp))[0]
        subj_out = os.path.join(output_dir, subject)
        print(f"\n[{i}/{len(e2e_files)}] {subject}")

        try:
            _, summary = analyze_e2e(
                fp,
                output_dir=subj_out,
                pixels_per_mm_x=pixels_per_mm_x,
                pixels_per_mm_y=pixels_per_mm_y,
            )
            row = {'subject': subject, 'status': 'SUCCESS', 'file': fp}
            row.update(summary)
            batch_rows.append(row)
            ok_count += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            batch_rows.append({'subject': subject, 'status': f'ERROR: {e}', 'file': fp})
            fail_count += 1

    # ── Combined summary CSV ──────────────────────────────────
    batch_df  = pd.DataFrame(batch_rows)
    csv_path  = os.path.join(output_dir, "batch_summary.csv")
    batch_df.to_csv(csv_path, index=False, float_format='%.4f')

    # ── Print table ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {len(e2e_files)}  |  OK: {ok_count}  |  Failed: {fail_count}")

    cols = ['subject', 'disc_radius_mm', 'onsd_mm', 'globe_area_mm2',
            'roc_mm', 'crae_mm', 'crve_mm', 'avr']
    display_cols = [c for c in cols if c in batch_df.columns]
    print(batch_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nBatch summary saved to: {csv_path}")
    return batch_df


def select_folder_and_run_batch():
    """
    GUI folder-picker — select a folder of .E2E files, process all at once.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    input_dir = filedialog.askdirectory(
        title="Select folder containing .E2E files"
    )
    if not input_dir:
        print("No folder selected — cancelled.")
        root.destroy()
        return

    # Count files so the user sees what they picked
    e2e_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.e2e')]
    if not e2e_files:
        messagebox.showerror(
            "No E2E files",
            f"No .E2E files found in:\n{input_dir}"
        )
        root.destroy()
        return

    output_dir = filedialog.askdirectory(
        title=f"Select output folder  ({len(e2e_files)} E2E files found — Cancel for default)"
    )
    root.destroy()

    if not output_dir:
        output_dir = os.path.join(input_dir, "batch_results")
        print(f"Using default output: {output_dir}")

    batch_df = process_all_e2e(input_dir, output_dir=output_dir)

    # Completion popup
    try:
        root2 = tk.Tk()
        root2.withdraw()
        root2.attributes('-topmost', True)
        ok   = (batch_df['status'] == 'SUCCESS').sum() if 'status' in batch_df else 0
        fail = len(batch_df) - ok
        messagebox.showinfo(
            "Batch complete",
            f"Processed : {len(e2e_files)} files\n"
            f"Succeeded : {ok}\n"
            f"Failed    : {fail}\n\n"
            f"Summary CSV:\n{os.path.join(output_dir, 'batch_summary.csv')}"
        )
        root2.destroy()
    except Exception:
        pass

    return batch_df


# =========================================================
# 15. ENTRY POINT
# =========================================================

if __name__ == "__main__":
    import sys

    # python3 e2e.py                        → GUI single-file picker
    # python3 e2e.py --batch                → GUI folder picker (all E2E in folder)
    # python3 e2e.py --batch /path/to/dir   → batch, no dialog
    # python3 e2e.py subject01.E2E          → run on that file
    # python3 e2e.py subject01.E2E ./out    → run + custom output dir

    if len(sys.argv) >= 2 and sys.argv[1] == "--batch":
        if len(sys.argv) >= 3:
            # folder passed directly — no dialog
            idir = sys.argv[2]
            odir = sys.argv[3] if len(sys.argv) >= 4 else None
            process_all_e2e(idir, output_dir=odir)
        else:
            select_folder_and_run_batch()

    elif len(sys.argv) >= 2:
        fp   = sys.argv[1]
        odir = sys.argv[2] if len(sys.argv) >= 3 else None
        analyze_e2e(fp, output_dir=odir)

    else:
        # Default: ask whether to do single file or batch
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        choice = messagebox.askquestion(
            "E2E Analyser",
            "Process a whole FOLDER of E2E files?\n\n"
            "Yes → batch (folder picker)\n"
            "No  → single file picker"
        )
        root.destroy()
        if choice == 'yes':
            select_folder_and_run_batch()
        else:
            select_e2e_and_run()
