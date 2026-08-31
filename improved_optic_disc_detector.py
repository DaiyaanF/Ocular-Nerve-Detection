import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class OpticDiscDetector:
    # Default calibration constants (used when no external calibration is supplied).
    #   Fundus  – 30° FOV, ~512 px across ~9.4 mm of retina  → ~54 px/mm
    #   B-scan  – standard ophthalmic ultrasound, ~77 µm/px  → ~13 px/mm
    DEFAULT_PPM_FUNDUS = 54.0   # pixels per mm
    DEFAULT_PPM_BSCAN  = 13.0   # pixels per mm

    def __init__(self, pixels_per_mm=None):
        self.debug = True
        self.image_type = None  # 'fundus' or 'bscan'
        # If supplied, overrides the per-type defaults for both image types.
        self.pixels_per_mm = pixels_per_mm
        # Set by _load_dicom() when DICOM spatial calibration tags are present.
        self._dicom_ppm = None

    @property
    def _ppm(self):
        """Return the effective pixels-per-mm for the current image type."""
        if self.pixels_per_mm is not None:
            return float(self.pixels_per_mm)
        if self._dicom_ppm is not None:
            return float(self._dicom_ppm)
        if self.image_type == 'bscan':
            return self.DEFAULT_PPM_BSCAN
        return self.DEFAULT_PPM_FUNDUS

    def px_to_mm(self, pixels):
        """Convert a pixel distance/length to millimetres."""
        if pixels is None:
            return None
        return pixels / self._ppm

    def px2_to_mm2(self, pixels_sq):
        """Convert a pixel² area to mm²."""
        if pixels_sq is None:
            return None
        return pixels_sq / (self._ppm ** 2)

    # ============================================================
    # IMAGE LOADING & TYPE DETECTION
    # ============================================================

    def _load_dicom(self, image_path):
        """
        Load a DICOM file and return a single grayscale frame.

        Handles:
          - Single-frame DICOM (shape H×W or H×W×3)
          - Multi-frame DICOM (shape N×H×W or N×H×W×3) — picks the frame
            whose mean brightness is closest to the overall median, which
            reliably avoids blank/artifact frames.
        """
        try:
            import pydicom
        except ImportError:
            raise ImportError(
                "pydicom is required to read DICOM files.\n"
                "Install with:  pip install pydicom pylibjpeg pylibjpeg-libjpeg"
            )

        ds  = pydicom.dcmread(image_path)
        arr = ds.pixel_array          # shape varies

        # Normalise to uint8
        if arr.dtype != np.uint8:
            arr = cv2.normalize(arr.astype(np.float32), None, 0, 255,
                                cv2.NORM_MINMAX).astype(np.uint8)

        # Multi-frame (N, H, W) or (N, H, W, C)
        if arr.ndim == 4:
            # colour multi-frame — pick best frame, then convert to grey
            frame_means = [arr[i].mean() for i in range(arr.shape[0])]
            med = float(np.median(frame_means))
            best = int(np.argmin([abs(m - med) for m in frame_means]))
            frame = arr[best]
        elif arr.ndim == 3:
            if arr.shape[2] in (3, 4):
                # Single colour frame
                frame = arr
            else:
                # (N, H, W) greyscale multi-frame
                frame_means = arr.mean(axis=(1, 2))
                med = float(np.median(frame_means))
                best = int(np.argmin(np.abs(frame_means - med)))
                frame = arr[best]
        else:
            frame = arr   # already 2D

        # Convert colour to greyscale
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Extract physical pixel spacing (px/mm).
        # Priority 1: SequenceOfUltrasoundRegions — standard for US DICOM
        ppm = None
        us_regions = getattr(ds, 'SequenceOfUltrasoundRegions', None)
        if us_regions:
            region = us_regions[0]
            dx = getattr(region, 'PhysicalDeltaX', None)
            units = getattr(region, 'PhysicalUnitsXDirection', 0)
            if dx is not None:
                dx_mm = float(dx)
                # units: 3 = cm → convert to mm
                if units == 3:
                    dx_mm *= 10.0
                # dx_mm is now mm/px → invert to px/mm
                if dx_mm > 0:
                    ppm = 1.0 / dx_mm
        # Priority 2: PixelSpacing tag (mm/px)
        if ppm is None:
            ps = getattr(ds, 'PixelSpacing', None)
            if ps is not None and len(ps) >= 2 and float(ps[0]) > 0:
                ppm = 1.0 / float(ps[0])
        self._dicom_ppm = ppm

        # Force image_type based on DICOM modality so auto-detection is not needed.
        # US = Ultrasound B-scan, OPT = OCT — both are B-scans for our pipeline.
        modality = getattr(ds, 'Modality', '').upper()
        if modality in ('US', 'OPT', 'OCT'):
            self.image_type = 'bscan'
        elif modality in ('CF', 'OP'):  # colour fundus / ophthalmic photography
            self.image_type = 'fundus'
        # else: leave None so detect_image_type() decides normally

        return frame

    def load_image(self, image_path):
        """Load and return a grayscale image (supports DICOM and standard formats)."""
        if image_path.lower().endswith('.dcm'):
            return self._load_dicom(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        return image

    def detect_image_type(self, image):
        """
        Detect whether the image is a fundus photograph or a B-scan.

        Decision uses three independent signals — any two out of three votes
        for B-scan wins.

        Signal 1 – Aspect ratio
            B-scans are typically 2:1 to 4:1 wide; fundus images are ~1:1.
            Score: +1 if w/h > 1.8

        Signal 2 – Vertical gradient dominance
            B-scans have horizontal tissue layers → large row-to-row (axis=0)
            intensity changes relative to column-to-column (axis=1) changes.
            np.diff(axis=0) measures how much intensity changes between rows.
            np.diff(axis=1) measures how much intensity changes between columns.
            Score: +1 if std(row-diffs) / std(col-diffs) > 1.05

        Signal 3 – Top-dark / bottom-bright pattern
            In a B-scan the vitreous (top) is near-black; the retinal wall and
            sclera at the bottom are bright echoes.  Fundus images are
            roughly uniformly lit.
            Score: +1 if mean(bottom-third) > mean(top-third) * 1.4
        """
        h, w = image.shape
        aspect_ratio = w / h

        score = 0

        # Signal 1: aspect ratio
        if aspect_ratio > 1.8:
            score += 1

        # Signal 2: row-to-row vs col-to-col gradient (corrected axis names)
        row_diff_std = np.std(np.diff(image.astype(float), axis=0))  # vertical gradient
        col_diff_std = np.std(np.diff(image.astype(float), axis=1))  # horizontal gradient
        if col_diff_std > 0 and row_diff_std / col_diff_std > 1.05:
            score += 1

        # Signal 3: dark vitreous on top, bright reflections on bottom
        top_mean = float(np.mean(image[:h // 3, :]))
        bot_mean = float(np.mean(image[2 * h // 3:, :]))
        if top_mean > 0 and bot_mean / top_mean > 1.4:
            score += 1

        if score >= 2:
            self.image_type = 'bscan'
            if self.debug:
                print(f"Detected B-scan (AR={aspect_ratio:.2f}, score={score}/3)")
            return 'bscan'

        self.image_type = 'fundus'
        if self.debug:
            print(f"Detected fundus (AR={aspect_ratio:.2f}, score={score}/3)")
        return 'fundus'

    # ============================================================
    # PREPROCESSING
    # ============================================================

    def preprocess_fundus_image(self, image):
        """Preprocessing for fundus images"""
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        return enhanced

    def preprocess_bscan_image(self, image):
        """Preprocessing specifically for B-scan images"""
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 4))
        enhanced = clahe.apply(denoised)
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_horizontal)
        enhanced = cv2.add(image, tophat)
        return enhanced

    def preprocess_image(self, image):
        """Adaptive preprocessing based on image type"""
        # If type was already set by DICOM modality, keep it; otherwise auto-detect.
        if self.image_type is None:
            self.detect_image_type(image)
        elif self.debug:
            h, w = image.shape
            print(f"Image type forced by DICOM modality: {self.image_type} "
                  f"(AR={w/h:.2f})")
        if self.image_type == 'bscan':
            return self.preprocess_bscan_image(image)
        else:
            return self.preprocess_fundus_image(image)

    # ============================================================
    # BRIGHT REGION DETECTION
    # ============================================================

    def find_bright_regions_fundus(self, image):
        mean_val = np.mean(image)
        std_val = np.std(image)
        methods = []

        thresh1 = min(mean_val + 2 * std_val, 255)
        _, binary1 = cv2.threshold(image, thresh1, 255, cv2.THRESH_BINARY)
        methods.append(("Statistical", binary1, thresh1))

        thresh2 = np.percentile(image, 98)
        _, binary2 = cv2.threshold(image, thresh2, 255, cv2.THRESH_BINARY)
        methods.append(("98th Percentile", binary2, thresh2))

        thresh3, binary3 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(("Otsu", binary3, thresh3))

        binary4 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        methods.append(("Adaptive", binary4, 0))

        return methods

    def find_bright_regions_bscan(self, image):
        mean_val = np.mean(image)
        std_val = np.std(image)
        methods = []

        thresh1 = min(mean_val + 1.5 * std_val, 255)
        _, binary1 = cv2.threshold(image, thresh1, 255, cv2.THRESH_BINARY)
        methods.append(("B-scan Statistical", binary1, thresh1))

        thresh2 = np.percentile(image, 95)
        _, binary2 = cv2.threshold(image, thresh2, 255, cv2.THRESH_BINARY)
        methods.append(("95th Percentile", binary2, thresh2))

        thresh3, binary3 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(("Otsu B-scan", binary3, thresh3))

        binary4 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 21, 2)
        methods.append(("Adaptive B-scan", binary4, 0))

        thresh5 = mean_val + 0.8 * std_val
        _, binary5 = cv2.threshold(image, thresh5, 255, cv2.THRESH_BINARY)
        methods.append(("B-scan Custom", binary5, thresh5))

        if self.debug:
            print(f"B-scan image stats: mean={mean_val:.1f}, std={std_val:.1f}")
            for name, _, thresh in methods:
                if thresh > 0:
                    print(f"  {name} threshold: {thresh:.1f}")

        return methods

    def find_bright_regions(self, image):
        if self.image_type == 'bscan':
            return self.find_bright_regions_bscan(image)
        else:
            return self.find_bright_regions_fundus(image)

    # ============================================================
    # CONTOUR ANALYSIS
    # ============================================================

    def analyze_contour_fundus(self, contour, original_image):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < 100 or perimeter == 0:
            return None

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        center = (int(center_x), int(center_y))

        mask = np.zeros(original_image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        mean_intensity = cv2.mean(original_image, mask=mask)[0]

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        return {
            'contour': contour,
            'center': center,
            'radius': int(radius),
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'intensity': mean_intensity,
            'solidity': solidity,
            'bbox': (x, y, w, h),
            'height': h,
            'width': w
        }

    def analyze_contour_bscan(self, contour, original_image):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < 50 or perimeter == 0:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        # aspect_ratio here is height/width — nerve head is tall, not wide
        aspect_ratio = float(h) / max(w, 1)

        # Accept both tall structures (h>w) and roughly square — the key
        # filter is applied in filter_candidates_bscan, not here.
        if aspect_ratio < 0.5:
            return None

        center = (int(x + w / 2), int(y + h / 2))

        mask = np.zeros(original_image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        mean_intensity = cv2.mean(original_image, mask=mask)[0]

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        return {
            'contour': contour,
            'center': center,
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'intensity': mean_intensity,
            'solidity': solidity,
            'bbox': (x, y, w, h),
            'height': h,
            'width': w,
            'nerve_head_height': h
        }

    def analyze_contour(self, contour, original_image):
        if self.image_type == 'bscan':
            return self.analyze_contour_bscan(contour, original_image)
        else:
            return self.analyze_contour_fundus(contour, original_image)

    # ============================================================
    # CANDIDATE FILTERING & SCORING
    # ============================================================

    def filter_candidates_fundus(self, candidates, image_shape):
        if not candidates:
            return []

        h, w = image_shape
        filtered = []

        for candidate in candidates:
            if 300 < candidate['area'] < 10000:
                if candidate['circularity'] > 0.2:
                    if 0.5 < candidate['aspect_ratio'] < 2.0:
                        if candidate['solidity'] > 0.7:
                            x, y = candidate['center']
                            margin = 50
                            if margin < x < w - margin and margin < y < h - margin:
                                filtered.append(candidate)

        def score_candidate(cand):
            brightness_score = cand['intensity'] / 255.0
            shape_score = cand['circularity']
            size_score = min(cand['area'] / 3000.0, 1.0)
            return 0.5 * brightness_score + 0.3 * shape_score + 0.2 * size_score

        filtered.sort(key=score_candidate, reverse=True)
        return filtered

    def filter_candidates_bscan(self, candidates, image_shape):
        if not candidates:
            return []

        img_h, img_w = image_shape
        filtered = []

        for candidate in candidates:
            area = candidate['area']
            h_cand = candidate['height']
            w_cand = candidate['width']
            cx, cy = candidate['center']

            # Area range: broader than fundus — nerve head can be large in B-scans
            if not (100 < area < 50000):
                continue
            # Must have some vertical extent
            if h_cand < 15:
                continue
            # Nerve head should not span more than 40 % of image height.
            # Anything taller is a large tissue mass or artifact, not the nerve.
            if h_cand > img_h * 0.40:
                continue
            # Must not be a horizontal sliver (e.g. the globe wall itself)
            if candidate['aspect_ratio'] < 0.4:
                continue
            # Must not hug the very edge — use proportional margins so the
            # filter scales with image size rather than a fixed 20 px.
            h_margin = max(20, int(img_h * 0.05))
            w_margin = max(20, int(img_w * 0.08))
            if not (w_margin < cx < img_w - w_margin and h_margin < cy < img_h - h_margin):
                continue
            # Nerve head center should not be in the top vitreous region (top 15 % of image)
            if cy < img_h * 0.15:
                continue
            # Exclude structures wider than 20 % of image width.
            # The previous 60 % limit allowed large tissue blobs to pass.
            if w_cand > img_w * 0.20:
                continue

            filtered.append(candidate)

        def score_bscan_candidate(cand):
            brightness_score = cand['intensity'] / 255.0
            # Reward structures sized like a real nerve head (5–20 % of image height).
            # Score peaks at 12 % of image height and falls off on both sides so that
            # oversized blobs no longer outscore compact, correctly-sized candidates.
            ideal_h = img_h * 0.12
            height_score = max(0.0, 1.0 - abs(cand['height'] - ideal_h) / ideal_h)
            vertical_score = min(cand['aspect_ratio'] / 2.0, 1.0)
            solidity_score = cand['solidity']
            return (0.40 * brightness_score
                    + 0.25 * height_score
                    + 0.20 * vertical_score
                    + 0.15 * solidity_score)

        filtered.sort(key=score_bscan_candidate, reverse=True)
        return filtered

    def filter_candidates(self, candidates, image_shape):
        if self.image_type == 'bscan':
            return self.filter_candidates_bscan(candidates, image_shape)
        else:
            return self.filter_candidates_fundus(candidates, image_shape)

    # ============================================================
    # OPTIC NERVE SHEATH DIAMETER (ONSD)
    # ============================================================

    @staticmethod
    def _gaussian_smooth_1d(arr, sigma=3):
        """Lightweight 1D Gaussian smoothing via convolution (no scipy needed)"""
        kernel_size = int(6 * sigma + 1) | 1  # ensure odd
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        return np.convolve(arr.astype(float), kernel, mode='same')

    def measure_onsd(self, image, nerve_candidate):
        """
        Measure Optic Nerve Sheath Diameter (ONSD).

        B-scan: scan horizontal profiles through the nerve centre and detect
        the full width of the echogenic nerve-plus-sheath complex using a
        40% above-background threshold (FWHM-style crossing).

        Fundus: ONSD cannot be seen directly; returns an anatomical estimate
        derived from the optic disc diameter (ODD × 1.4, based on the known
        anatomical ratio between disc diameter and nerve sheath width).

        Units: pixels.  Divide by self.pixels_per_mm for mm, if calibrated.
        """
        if self.image_type == 'bscan':
            return self._measure_onsd_bscan(image, nerve_candidate)
        else:
            return self._measure_onsd_fundus(nerve_candidate)

    def _measure_onsd_bscan(self, image, nerve_candidate):
        bbox = nerve_candidate['bbox']
        x, w, h = bbox[0], bbox[2], bbox[3]
        cy = nerve_candidate['center'][1]
        img_h, img_w = image.shape

        # Sample three rows: slightly above centre, centre, slightly below
        sample_rows = [
            max(0, int(cy - h * 0.1)),
            max(0, cy),
            min(img_h - 1, int(cy + h * 0.1))
        ]

        widths = []
        for row in sample_rows:
            search_left = max(0, x - w)
            search_right = min(img_w, x + 2 * w)
            profile = image[row, search_left:search_right].astype(float)
            if len(profile) < 10:
                continue

            profile_smooth = self._gaussian_smooth_1d(profile, sigma=3)
            bg_left = float(np.median(profile_smooth[:max(1, len(profile_smooth) // 5)]))
            bg_right = float(np.median(profile_smooth[max(1, 4 * len(profile_smooth) // 5):]))
            background = (bg_left + bg_right) / 2.0
            peak = float(np.max(profile_smooth))

            if peak <= background + 5:
                continue

            threshold = background + 0.40 * (peak - background)
            above = profile_smooth > threshold
            if not np.any(above):
                continue

            indices = np.where(above)[0]
            width_px = int(indices[-1] - indices[0])
            if width_px > 5:
                widths.append(width_px)

        if widths:
            onsd = float(np.median(widths))
        else:
            # Fallback: bounding-box width + estimated sheath margin (15 % each side)
            onsd = float(w) * 1.30

        if self.debug:
            print(f"ONSD (B-scan): {onsd:.1f} px"
                  + (f"  /  {onsd / self.pixels_per_mm:.2f} mm" if self.pixels_per_mm else ""))
        return onsd

    def _measure_onsd_fundus(self, nerve_candidate):
        disc_diameter = nerve_candidate['radius'] * 2
        onsd = disc_diameter * 1.4
        if self.debug:
            print(f"ONSD (fundus estimate): {onsd:.1f} px")
        return float(onsd)

    # ============================================================
    # POSTERIOR GLOBE — CROSS-SECTIONAL AREA & RADIUS OF CURVATURE
    # ============================================================

    def measure_posterior_globe(self, image, nerve_candidate=None):  # noqa: ARG002
        """
        Detect the posterior globe outline and return:
          - globe_center       : (x, y) in pixels
          - globe_radius       : best-fit circle radius in pixels
          - cross_sectional_area : π r²  in pixels²
          - radius_of_curvature  : same as globe_radius (posterior surface curvature)
          - method             : detection strategy used

        B-scan: Hough circle on the blurred image; the largest detected circle is
        the globe wall.  Falls back to an image-height-based estimate.

        Fundus: Hough circle on the illuminated field boundary (the circular
        bright area that bounds a fundus image).  Falls back to min(h, w)/2.
        """
        if self.image_type == 'bscan':
            return self._measure_globe_bscan(image)
        else:
            return self._measure_globe_fundus(image)

    def _measure_globe_bscan(self, image):
        img_h, img_w = image.shape
        blurred = cv2.GaussianBlur(image, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=img_h // 3,
            param1=50,
            param2=30,
            minRadius=img_h // 6,
            maxRadius=min(img_h, img_w) // 2
        )

        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            # Largest circle = globe
            cx, cy, r = sorted(circles, key=lambda c: c[2], reverse=True)[0]
            area = np.pi * r ** 2
            if self.debug:
                print(f"Posterior globe (Hough): center=({cx},{cy}), r={r}px, area={area:.0f}px²")
            return {
                'globe_center': (int(cx), int(cy)),
                'globe_radius': float(r),
                'cross_sectional_area': float(area),
                'radius_of_curvature': float(r),
                'method': 'hough_circle'
            }

        # Fallback: globe typically spans ~70 % of image height
        r = img_h * 0.35
        area = np.pi * r ** 2
        if self.debug:
            print(f"Posterior globe (fallback estimate): r={r:.1f}px")
        return {
            'globe_center': (img_w // 2, img_h // 2),
            'globe_radius': r,
            'cross_sectional_area': float(area),
            'radius_of_curvature': r,
            'method': 'estimated'
        }

    def _measure_globe_fundus(self, image):
        # The posterior globe wall is NOT visible in a fundus photograph —
        # only the inner retinal surface is imaged.  Measuring the camera
        # FOV boundary produces numbers that change with zoom/camera model,
        # creating spurious pre/post deltas.  Return None for both metrics
        # so they are excluded from the CSV rather than silently wrong.
        img_h, img_w = image.shape
        if self.debug:
            print("Posterior globe (fundus): not measurable from fundus image — skipped")
        return {
            'globe_center': (img_w // 2, img_h // 2),
            'globe_radius': None,
            'cross_sectional_area': None,
            'radius_of_curvature': None,
            'method': 'not_applicable'
        }

    # ============================================================
    # VESSEL DETECTION (for fundus images)
    # ============================================================

    def enhance_vessels(self, image):
        """
        Enhance dark retinal vessels via multi-scale morphological black-hat
        transform followed by CLAHE.  Vessels appear dark against the bright
        fundus background, so black-hat extracts dark details.
        """
        vessel_enhanced = np.zeros(image.shape, dtype=float)
        for ks in [3, 5, 7, 9]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
            blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            vessel_enhanced += blackhat.astype(float)

        vessel_enhanced /= 4.0
        vessel_enhanced = np.clip(vessel_enhanced, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(vessel_enhanced)

    def detect_vessels(self, image):
        """
        Segment retinal vessels.
        Returns (vessel_mask [uint8 binary], vessel_enhanced [uint8]).
        """
        enhanced = self.enhance_vessels(image)
        mean_val = np.mean(enhanced)
        std_val = np.std(enhanced)
        thresh_val = min(mean_val + 1.0 * std_val, 200)
        _, vessel_mask = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, kernel)
        return vessel_mask, enhanced

    # ============================================================
    # CRAE / CRVE / AVR
    # ============================================================

    @staticmethod
    def _knudtson_combine(diameters, coeff):
        """
        Iterative Knudtson vessel-equivalent formula:
            W_new = coeff * sqrt(d_larger² + d_smaller²)
        Pairs are sorted largest-first; odd remainders carry forward unchanged.
        """
        d = sorted(diameters, reverse=True)
        while len(d) > 1:
            new_d = []
            for i in range(0, len(d) - 1, 2):
                new_d.append(coeff * np.sqrt(d[i] ** 2 + d[i + 1] ** 2))
            if len(d) % 2 == 1:
                new_d.append(d[-1])
            d = sorted(new_d, reverse=True)
        return float(d[0])

    def _measure_vessel_calibers_in_zone(self, vessel_mask, disc_center, disc_radius, image):
        """
        Measure vessel calibers inside the peripapillary annulus
        (0.5 × disc_radius  to  1.5 × disc_radius from the disc centre).

        Each connected component in the annular zone gets an estimated caliber:
            caliber ≈ component_area / max(component_width, component_height)

        Vessels are split into arteries (above-median intensity → brighter)
        and veins (below-median intensity → darker).
        """
        img_h, img_w = vessel_mask.shape
        cx, cy = disc_center
        inner_r = int(disc_radius * 0.5)
        outer_r = int(disc_radius * 1.5)

        yy, xx = np.ogrid[:img_h, :img_w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        annular = ((dist >= inner_r) & (dist <= outer_r)).astype(np.uint8) * 255
        zone_vessels = cv2.bitwise_and(vessel_mask, annular)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(zone_vessels)

        vessel_data = []
        for i in range(1, num_labels):
            comp_area = stats[i, cv2.CC_STAT_AREA]
            if comp_area < 10:
                continue
            comp_w = max(stats[i, cv2.CC_STAT_WIDTH], 1)
            comp_h = max(stats[i, cv2.CC_STAT_HEIGHT], 1)
            caliber = comp_area / float(max(comp_w, comp_h))
            if caliber < 2:
                continue
            comp_mask = (labels == i).astype(np.uint8) * 255
            mean_int = cv2.mean(image, mask=comp_mask)[0]
            vessel_data.append((caliber, mean_int))

        if not vessel_data:
            return [], []

        # Take the 12 largest caliber vessels for the formula
        vessel_data.sort(key=lambda v: v[0], reverse=True)
        top = vessel_data[:12]

        intensities = [v[1] for v in top]
        median_int = float(np.median(intensities))

        arteries = [c for c, intv in top if intv >= median_int]
        veins    = [c for c, intv in top if intv < median_int]

        return arteries, veins

    def calculate_vascular_metrics(self, image, disc_center, disc_radius):
        """
        Compute CRAE, CRVE, and AVR for a fundus image.

        CRAE (Central Retinal Arteriole Equivalent): Knudtson coeff = 0.88
        CRVE (Central Retinal Venule Equivalent):    Knudtson coeff = 0.95
        AVR  (Arteriove nous Ratio):                 CRAE / CRVE
            Normal range: 0.67–0.75.  Values < 0.67 suggest arteriolar narrowing.

        All values in pixels; divide by self.pixels_per_mm for mm.
        """
        vessel_mask, _ = self.detect_vessels(image)
        arteries, veins = self._measure_vessel_calibers_in_zone(
            vessel_mask, disc_center, disc_radius, image
        )

        crae = self._knudtson_combine(arteries, 0.88) if len(arteries) >= 2 else (arteries[0] if arteries else None)
        crve = self._knudtson_combine(veins,    0.95) if len(veins)    >= 2 else (veins[0]    if veins    else None)

        # Physiological constraint: arteries are always narrower than veins (AVR < 1).
        # If the classifier flipped them, swap and recompute.
        if crae is not None and crve is not None and crae > crve:
            arteries, veins = veins, arteries
            crae = self._knudtson_combine(arteries, 0.88) if len(arteries) >= 2 else (arteries[0] if arteries else None)
            crve = self._knudtson_combine(veins,    0.95) if len(veins)    >= 2 else (veins[0]    if veins    else None)
            if self.debug:
                print("  Artery/vein labels swapped to enforce CRAE < CRVE")

        avr = (crae / crve) if (crae and crve and crve > 0 and crae < crve) else None

        if self.debug:
            print(f"Artery calibers ({len(arteries)}): {[f'{c:.1f}' for c in arteries]}")
            print(f"Vein   calibers ({len(veins)}):   {[f'{c:.1f}' for c in veins]}")
            print(f"CRAE: {crae:.2f} px" if crae else "CRAE: N/A")
            print(f"CRVE: {crve:.2f} px" if crve else "CRVE: N/A")
            print(f"AVR : {avr:.3f}"     if avr  else "AVR : N/A")

        return {
            'crae': crae,
            'crve': crve,
            'avr': avr,
            'artery_calibers': arteries,
            'vein_calibers': veins,
            'vessel_mask': vessel_mask
        }

    # ============================================================
    # MAIN DETECTION
    # ============================================================

    def detect_optic_disc(self, image_path):
        """
        Main detection pipeline.

        Returns
        -------
        candidates : list of dicts, sorted best-first
        original   : grayscale ndarray
        processed  : preprocessed grayscale ndarray
        measurements : dict with keys:
            optic_disc_radius      (fundus only, px)
            nerve_head_height      (B-scan only, px)
            nerve_head_width       (B-scan only, px)
            onsd                   (px)
            posterior_globe        (dict with globe_center, globe_radius, …)
            posterior_globe_area   (px²)
            radius_of_curvature    (px)
            crae                   (px, fundus only)
            crve                   (px, fundus only)
            avr                    (dimensionless, fundus only)
            vessel_mask            (ndarray, fundus only)
        """
        # Reset per-image state so a previous file's DICOM modality / scale
        # never bleeds into the next file.
        self.image_type  = None
        self._dicom_ppm  = None

        image = self.load_image(image_path)
        processed = self.preprocess_image(image)

        if self.debug:
            print(f"\nProcessing: {os.path.basename(image_path)}")
            print(f"Image type: {self.image_type}")
            print(f"Image shape: {image.shape}")
            print(f"Intensity range: {image.min()} – {image.max()}")

        threshold_methods = self.find_bright_regions(processed)
        all_candidates = []

        for method_name, binary_mask, _ in threshold_methods:
            if self.image_type == 'bscan':
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if self.debug:
                print(f"  {method_name}: {len(contours)} contours")

            for contour in contours:
                candidate = self.analyze_contour(contour, image)
                if candidate:
                    candidate['method'] = method_name
                    all_candidates.append(candidate)

        good_candidates = self.filter_candidates(all_candidates, image.shape)

        if self.debug:
            print(f"Good candidates: {len(good_candidates)}")
            for i, cand in enumerate(good_candidates[:3]):
                if self.image_type == 'bscan':
                    print(f"  {i+1}. center={cand['center']}, h={cand['height']}, "
                          f"w={cand['width']}, intensity={cand['intensity']:.1f}")
                else:
                    print(f"  {i+1}. center={cand['center']}, radius={cand.get('radius','N/A')}, "
                          f"intensity={cand['intensity']:.1f}, circ={cand.get('circularity',0):.2f}")

        # ---- Extended measurements ----------------------------------------
        measurements = {}

        if good_candidates:
            best = good_candidates[0]

            # 1. Optic Nerve Sheath Diameter
            onsd_px = self.measure_onsd(image, best)
            measurements['onsd']    = onsd_px
            measurements['onsd_mm'] = self.px_to_mm(onsd_px)

            # 2. Posterior globe cross-sectional area & radius of curvature
            globe = self.measure_posterior_globe(image, best)
            measurements['posterior_globe']          = globe
            measurements['posterior_globe_area']     = globe['cross_sectional_area']
            measurements['posterior_globe_area_mm2'] = self.px2_to_mm2(globe['cross_sectional_area'])
            measurements['radius_of_curvature']      = globe['radius_of_curvature']
            measurements['radius_of_curvature_mm']   = self.px_to_mm(globe['radius_of_curvature'])

            if self.image_type == 'fundus':
                # 3a. Optic disc radius
                disc_r = float(best.get('radius', 0))
                measurements['optic_disc_radius']    = disc_r
                measurements['optic_disc_radius_mm'] = self.px_to_mm(disc_r)

                # 4. Vascular metrics (CRAE / CRVE / AVR)
                vascular = self.calculate_vascular_metrics(
                    image, best['center'], best.get('radius', 30)
                )
                measurements.update(vascular)
                measurements['crae_mm'] = self.px_to_mm(vascular.get('crae'))
                measurements['crve_mm'] = self.px_to_mm(vascular.get('crve'))
                # AVR is dimensionless — no unit conversion needed
            else:
                # 3b. Nerve head dimensions
                nh = best.get('height')
                nw = best.get('width')
                measurements['nerve_head_height']    = nh
                measurements['nerve_head_height_mm'] = self.px_to_mm(nh)
                measurements['nerve_head_width']     = nw
                measurements['nerve_head_width_mm']  = self.px_to_mm(nw)

        return good_candidates, image, processed, measurements

    # ============================================================
    # VISUALISATION
    # ============================================================

    def visualize_results(self, candidates, original_image, processed_image,
                          measurements=None, save_path=None):
        """
        2×3 figure:
          [0,0] Original          [0,1] Preprocessed      [0,2] Threshold view
          [1,0] Detection overlay [1,1] Vessel map         [1,2] Measurements table
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title(f'Original Image ({self.image_type.upper()})')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(processed_image, cmap='gray')
        axes[0, 1].set_title('Preprocessed Image')
        axes[0, 1].axis('off')

        result = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        if candidates:
            best = candidates[0]
            center = best['center']

            if self.image_type == 'bscan':
                x, y, w, h = best['bbox']
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(result, center, 3, (0, 0, 255), -1)
                height = best['height']
                cv2.putText(result, "Nerve Head",
                            (center[0] - 40, max(5, center[1] - h // 2 - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(result, f"H: {height}px",
                            (center[0] - 40, max(5, center[1] - h // 2 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # ONSD measurement line
                if measurements and measurements.get('onsd'):
                    onsd = measurements['onsd']
                    half = int(onsd / 2)
                    cv2.line(result,
                             (center[0] - half, center[1]),
                             (center[0] + half, center[1]),
                             (0, 255, 255), 2)
                    cv2.putText(result, f"ONSD: {onsd:.1f}px",
                                (center[0] - half, max(5, center[1] - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                # Posterior globe circle
                if measurements and measurements.get('posterior_globe'):
                    globe = measurements['posterior_globe']
                    if globe.get('globe_radius') is not None:
                        gc = globe['globe_center']
                        gr = int(globe['globe_radius'])
                        cv2.circle(result, gc, gr, (255, 165, 0), 2)

            else:
                radius = best.get('radius', 20)
                cv2.circle(result, center, radius, (0, 255, 0), 3)
                cv2.circle(result, center, 3, (0, 0, 255), -1)
                cv2.drawContours(result, [best['contour']], -1, (0, 255, 0), 2)
                cv2.putText(result, "Optic Disc",
                            (center[0] - 40, max(5, center[1] - radius - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if measurements and measurements.get('onsd'):
                    onsd = measurements['onsd']
                    cv2.putText(result, f"Est. ONSD: {onsd:.1f}px",
                                (center[0] - 60, center[1] + radius + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Overlay vessel map in magenta
                if measurements and measurements.get('vessel_mask') is not None:
                    vm = measurements['vessel_mask']
                    result[vm > 0] = [200, 0, 180]

                # Posterior globe boundary
                if measurements and measurements.get('posterior_globe'):
                    globe = measurements['posterior_globe']
                    if globe.get('globe_radius') is not None:
                        gc = globe['globe_center']
                        gr = int(globe['globe_radius'])
                        cv2.circle(result, gc, gr, (255, 165, 0), 2)

        axes[1, 0].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f'Detection ({self.image_type.upper()})')
        axes[1, 0].axis('off')

        # Threshold visualisation
        if candidates:
            pct = 95 if self.image_type == 'bscan' else 98
            tv = np.percentile(processed_image, pct)
            _, thresh_img = cv2.threshold(processed_image, tv, 255, cv2.THRESH_BINARY)
            axes[0, 2].imshow(thresh_img, cmap='gray')
            axes[0, 2].set_title('Threshold View')
        else:
            axes[0, 2].text(0.5, 0.5, 'No Detection', transform=axes[0, 2].transAxes,
                            ha='center', va='center', fontsize=16, color='red')
        axes[0, 2].axis('off')

        # Vessel map panel (fundus) or plain image (B-scan)
        if (self.image_type == 'fundus'
                and measurements
                and measurements.get('vessel_mask') is not None):
            axes[1, 1].imshow(measurements['vessel_mask'], cmap='hot')
            axes[1, 1].set_title('Vessel Segmentation Map')
        else:
            axes[1, 1].imshow(original_image, cmap='gray')
            axes[1, 1].set_title('Original (no vessel map)')
        axes[1, 1].axis('off')

        # Measurements text panel
        axes[1, 2].axis('off')
        if measurements:
            ppm = self._ppm

            def fmt(px_val, mm_val, unit='mm', dp_mm=2):
                if px_val is None:
                    return "N/A"
                mm_s = f"{mm_val:.{dp_mm}f} {unit}" if mm_val is not None else "? mm"
                return f"{px_val:.1f} px  ({mm_s})"

            def fmt_area(px2_val, mm2_val):
                if px2_val is None:
                    return "N/A"
                mm2_s = f"{mm2_val:.2f} mm²" if mm2_val is not None else "? mm²"
                return f"{px2_val:.0f} px²  ({mm2_s})"

            cal_src = "user" if self.pixels_per_mm else ("B-scan default" if self.image_type == 'bscan' else "fundus default")
            lines = [
                "MEASUREMENTS",
                "=" * 36,
                f"Scale: {ppm:.1f} px/mm  [{cal_src}]",
                "",
            ]

            if self.image_type == 'fundus':
                lines.append(f"Optic Disc Radius  : {fmt(measurements.get('optic_disc_radius'), measurements.get('optic_disc_radius_mm'))}")
            else:
                lines.append(f"Nerve Head Height  : {fmt(measurements.get('nerve_head_height'), measurements.get('nerve_head_height_mm'))}")
                lines.append(f"Nerve Head Width   : {fmt(measurements.get('nerve_head_width'), measurements.get('nerve_head_width_mm'))}")

            lines.append("")
            lines.append(f"ONSD               : {fmt(measurements.get('onsd'), measurements.get('onsd_mm'))}")
            lines.append(f"Post. Globe Area   : {fmt_area(measurements.get('posterior_globe_area'), measurements.get('posterior_globe_area_mm2'))}")
            lines.append(f"Radius of Curv.    : {fmt(measurements.get('radius_of_curvature'), measurements.get('radius_of_curvature_mm'))}")

            if self.image_type == 'fundus':
                crae = measurements.get('crae')
                crve = measurements.get('crve')
                avr  = measurements.get('avr')
                lines.append("")
                lines.append(f"CRAE               : {fmt(crae, measurements.get('crae_mm'))}")
                lines.append(f"CRVE               : {fmt(crve, measurements.get('crve_mm'))}")
                avr_s = f"{avr:.3f}" if avr else "N/A"
                norm  = " (normal)" if avr and 0.67 <= avr <= 0.75 else \
                        " (narrow)"  if avr and avr < 0.67 else \
                        " (wide v.)" if avr else ""
                lines.append(f"AVR                : {avr_s}{norm}")

            axes[1, 2].text(0.05, 0.95, "\n".join(lines),
                            transform=axes[1, 2].transAxes,
                            fontsize=9, verticalalignment='top',
                            fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 2].set_title('Measurements')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            cv2.imwrite(save_path.replace('.png', '_result.png'), result)

        plt.show()
        return result


# ================================================================
# BATCH PROCESSING
# ================================================================

def process_all_images(input_dir=None, output_dir=None, pixels_per_mm=None, recursive=True):
    """
    Process all images under input_dir and save per-image PNG results + a CSV
    summary to output_dir.

    Parameters
    ----------
    input_dir     : root folder to search (prompted if None)
    output_dir    : where to write results (auto-created; defaults to a
                    subfolder next to input_dir)
    pixels_per_mm : override scale factor; None = use per-type defaults
    recursive     : if True (default) walk all subdirectories; if False only
                    look in the immediate input_dir
    """
    EXTS = {'.pgm', '.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp', '.dcm'}

    detector = OpticDiscDetector(pixels_per_mm=pixels_per_mm)

    if input_dir is None:
        input_dir = os.environ.get(
            'OPTIC_INPUT_DIR',
            os.path.join(os.path.dirname(__file__), 'images')
        )

    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    # Suffixes added by this tool's own visualise_results / process_all_images.
    # Skipping them prevents the detector from re-processing its own output images.
    OUTPUT_SUFFIXES = ('_detection', '_detection_result')

    def _is_own_output(fname):
        stem = os.path.splitext(fname)[0]
        return any(stem.endswith(s) for s in OUTPUT_SUFFIXES)

    # Collect (absolute_path, relative_path) pairs
    image_files = []
    if recursive:
        for root, _dirs, files in os.walk(input_dir):
            for fname in sorted(files):
                if os.path.splitext(fname.lower())[1] in EXTS and not _is_own_output(fname):
                    abs_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(abs_path, input_dir)
                    image_files.append((abs_path, rel_path))
    else:
        for fname in sorted(os.listdir(input_dir)):
            if os.path.splitext(fname.lower())[1] in EXTS and not _is_own_output(fname):
                abs_path = os.path.join(input_dir, fname)
                image_files.append((abs_path, fname))

    if not image_files:
        print("No image files found in the directory")
        return

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'optic_disc_results')

    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(image_files)} images to process")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    successful = 0
    failed = 0
    results_summary = []
    fundus_count = 0
    bscan_count = 0

    for i, (image_path, rel_path) in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        # Extract subject and condition from path components, e.g.
        # Flight/TRISH-4449/scan.dcm  → condition=Flight, subject=TRISH-4449
        parts = rel_path.replace('\\', '/').split('/')
        condition = parts[0] if len(parts) >= 3 else ''
        subject   = parts[1] if len(parts) >= 3 else (parts[0] if len(parts) == 2 else '')
        print(f"\n[{i}/{len(image_files)}] {rel_path}")

        try:
            candidates, original, processed, measurements = detector.detect_optic_disc(image_path)

            base = os.path.splitext(filename)[0]
            save_path = os.path.join(output_dir, f"{base}_detection.png")

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes[0, 0].imshow(original, cmap='gray')
            axes[0, 0].set_title(f'Original ({detector.image_type.upper()})')
            axes[0, 0].axis('off')
            axes[0, 1].imshow(processed, cmap='gray')
            axes[0, 1].set_title('Preprocessed')
            axes[0, 1].axis('off')

            result = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

            if candidates:
                best = candidates[0]
                center = best['center']

                if detector.image_type == 'bscan':
                    bscan_count += 1
                    x, y, w, h = best['bbox']
                    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(result, center, 3, (0, 0, 255), -1)
                    cv2.putText(result, f"H:{best['height']}px",
                                (center[0] - 30, max(5, center[1] - h // 2 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    if measurements.get('onsd'):
                        half = int(measurements['onsd'] / 2)
                        cv2.line(result,
                                 (center[0] - half, center[1]),
                                 (center[0] + half, center[1]),
                                 (0, 255, 255), 2)
                    if measurements.get('posterior_globe'):
                        if measurements['posterior_globe'].get('globe_radius') is not None:
                            gc = measurements['posterior_globe']['globe_center']
                            gr = int(measurements['posterior_globe']['globe_radius'])
                            cv2.circle(result, gc, gr, (255, 165, 0), 2)
                else:
                    fundus_count += 1
                    radius = best.get('radius', 20)
                    cv2.circle(result, center, radius, (0, 255, 0), 3)
                    cv2.circle(result, center, 3, (0, 0, 255), -1)
                    cv2.drawContours(result, [best['contour']], -1, (0, 255, 0), 2)
                    if measurements.get('vessel_mask') is not None:
                        result[measurements['vessel_mask'] > 0] = [200, 0, 180]
                    if measurements.get('posterior_globe'):
                        if measurements['posterior_globe'].get('globe_radius') is not None:
                            gc = measurements['posterior_globe']['globe_center']
                            gr = int(measurements['posterior_globe']['globe_radius'])
                            cv2.circle(result, gc, gr, (255, 165, 0), 2)

                axes[1, 0].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                axes[1, 0].set_title(f'Detection ({detector.image_type.upper()})')
                axes[1, 0].axis('off')

                pct = 95 if detector.image_type == 'bscan' else 98
                tv = np.percentile(processed, pct)
                _, thresh_img = cv2.threshold(processed, tv, 255, cv2.THRESH_BINARY)
                axes[0, 2].imshow(thresh_img, cmap='gray')
                axes[0, 2].set_title('Threshold View')
                axes[0, 2].axis('off')

                if (detector.image_type == 'fundus'
                        and measurements.get('vessel_mask') is not None):
                    axes[1, 1].imshow(measurements['vessel_mask'], cmap='hot')
                    axes[1, 1].set_title('Vessel Map')
                else:
                    axes[1, 1].imshow(original, cmap='gray')
                    axes[1, 1].set_title('Image')
                axes[1, 1].axis('off')

                # Measurements text — all values in mm
                axes[1, 2].axis('off')

                def _mm(val, dp=3):
                    return f"{val:.{dp}f} mm" if val is not None else "N/A"

                def _mm2(val, dp=2):
                    return f"{val:.{dp}f} mm²" if val is not None else "N/A"

                ppm = detector._ppm
                cal_src = "user" if pixels_per_mm else (
                    "B-scan default" if detector.image_type == 'bscan' else "fundus default"
                )
                lines = [
                    "MEASUREMENTS",
                    "=" * 30,
                    f"Scale: {ppm:.1f} px/mm  [{cal_src}]",
                    "",
                ]
                if detector.image_type == 'fundus':
                    lines.append(f"Disc Radius : {_mm(measurements.get('optic_disc_radius_mm'))}")
                else:
                    lines.append(f"Nerve H     : {_mm(measurements.get('nerve_head_height_mm'))}")
                    lines.append(f"Nerve W     : {_mm(measurements.get('nerve_head_width_mm'))}")
                lines.append(f"ONSD        : {_mm(measurements.get('onsd_mm'))}")
                lines.append(f"Globe Area  : {_mm2(measurements.get('posterior_globe_area_mm2'))}")
                lines.append(f"RoC         : {_mm(measurements.get('radius_of_curvature_mm'))}")
                if detector.image_type == 'fundus':
                    avr = measurements.get('avr')
                    lines.append(f"CRAE        : {_mm(measurements.get('crae_mm'), 4)}")
                    lines.append(f"CRVE        : {_mm(measurements.get('crve_mm'), 4)}")
                    avr_norm = (" (normal)" if avr and 0.67 <= avr <= 0.75 else
                                " (narrow)"  if avr and avr < 0.67 else
                                " (wide v.)" if avr else "")
                    lines.append(f"AVR         : {f'{avr:.3f}{avr_norm}' if avr else 'N/A'}")
                axes[1, 2].text(0.05, 0.95, "\n".join(lines),
                                transform=axes[1, 2].transAxes,
                                fontsize=9, verticalalignment='top',
                                fontfamily='monospace',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                axes[1, 2].set_title('Measurements')

                successful += 1
                ppm = detector._ppm
                results_summary.append({
                    'filename': rel_path,
                    'condition': condition,
                    'subject': subject,
                    'status': 'SUCCESS',
                    'image_type': detector.image_type,
                    'pixels_per_mm': ppm,
                    'center_x': center[0],
                    'center_y': center[1],
                    'intensity': best['intensity'],
                    'area': best['area'],
                    'optic_disc_radius_mm': measurements.get('optic_disc_radius_mm', ''),
                    'nerve_head_height_mm': measurements.get('nerve_head_height_mm', ''),
                    'nerve_head_width_mm':  measurements.get('nerve_head_width_mm', ''),
                    'onsd_mm': measurements.get('onsd_mm', ''),
                    'posterior_globe_area_mm2': measurements.get('posterior_globe_area_mm2', ''),
                    'radius_of_curvature_mm': measurements.get('radius_of_curvature_mm', ''),
                    'crae_mm': measurements.get('crae_mm', ''),
                    'crve_mm': measurements.get('crve_mm', ''),
                    'avr': measurements.get('avr', ''),
                })
                onsd = measurements.get('onsd_mm')
                area = measurements.get('posterior_globe_area_mm2')
                roc  = measurements.get('radius_of_curvature_mm')
                print(f"  DETECTED ({detector.image_type})  [scale: {ppm:.1f} px/mm]")
                print(f"    ONSD={onsd:.2f}mm  Globe={area:.2f}mm²  RoC={roc:.2f}mm" if all([onsd, area, roc]) else "")
                if detector.image_type == 'fundus':
                    crae = measurements.get('crae_mm')
                    crve = measurements.get('crve_mm')
                    avr  = measurements.get('avr')
                    if crae and crve:
                        print(f"    CRAE={crae:.3f}mm  CRVE={crve:.3f}mm  AVR={avr:.3f}" if avr else "")

            else:
                for ax in [axes[1, 0], axes[0, 2], axes[1, 1], axes[1, 2]]:
                    ax.text(0.5, 0.5, 'No Detection', transform=ax.transAxes,
                            ha='center', va='center', fontsize=14, color='red')
                    ax.axis('off')
                failed += 1
                results_summary.append({
                    'filename': rel_path, 'condition': condition, 'subject': subject,
                    'status': 'FAILED', 'image_type': detector.image_type
                })
                print(f"  NO DETECTION ({detector.image_type})")

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            cv2.imwrite(save_path.replace('.png', '_result.png'), result)
            plt.close()

        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
            results_summary.append({
                'filename': rel_path, 'condition': condition, 'subject': subject,
                'status': 'ERROR', 'error': str(e)
            })

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total: {len(image_files)}  |  Fundus: {fundus_count}  |  B-scan: {bscan_count}")
    print(f"Success: {successful}  |  Failed: {failed}  |  "
          f"Rate: {successful / max(len(image_files), 1) * 100:.1f}%")

    csv_path = os.path.join(output_dir, "detection_results.csv")
    N_COLS = 13  # data columns after Condition,Subject,Filename,Status,Image_Type,Pixels_Per_MM
    with open(csv_path, 'w') as f:
        f.write(
            "Condition,Subject,Filename,Status,Image_Type,Pixels_Per_MM,"
            "ONSD_mm,"
            "Posterior_Globe_Area_mm2,"
            "Radius_of_Curvature_mm,"
            "CRAE_mm,"
            "CRVE_mm,"
            "AVR,"
            "Optic_Disc_Radius_mm,"
            "Nerve_Head_Height_mm,"
            "Nerve_Head_Width_mm\n"
        )
        for r in results_summary:
            def f2(v, dp=3):
                if v == '' or v is None:
                    return ''
                if isinstance(v, float):
                    return f"{v:.{dp}f}"
                return str(v)

            cond = r.get('condition', '')
            subj = r.get('subject', '')
            if r['status'] == 'SUCCESS':
                f.write(
                    f"{cond},{subj},{r['filename']},SUCCESS,{r['image_type']},{r['pixels_per_mm']:.1f},"
                    f"{f2(r['onsd_mm'])},"
                    f"{f2(r['posterior_globe_area_mm2'], 2)},"
                    f"{f2(r['radius_of_curvature_mm'])},"
                    f"{f2(r['crae_mm'])},"
                    f"{f2(r['crve_mm'])},"
                    f"{f2(r['avr'])},"
                    f"{f2(r['optic_disc_radius_mm'])},"
                    f"{f2(r['nerve_head_height_mm'])},"
                    f"{f2(r['nerve_head_width_mm'])}\n"
                )
            else:
                f.write(f"{cond},{subj},{r['filename']},{r['status']},{r.get('image_type','unknown')}"
                        + "," * N_COLS + "\n")

    print(f"\nResults saved to: {csv_path}")
    return results_summary


# ================================================================
# SINGLE-IMAGE TEST
# ================================================================

def test_detection(input_dir=None):
    """Test with the first image found in input_dir"""
    detector = OpticDiscDetector()

    if input_dir is None:
        input_dir = os.environ.get(
            'OPTIC_INPUT_DIR',
            os.path.join(os.path.dirname(__file__), 'images')
        )

    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    image_files = []
    for ext in ['.pgm', '.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp', '.dcm']:
        image_files.extend(
            [f for f in os.listdir(input_dir) if f.lower().endswith(ext)]
        )

    if not image_files:
        print("No image files found in the directory")
        return

    test_image = os.path.join(input_dir, image_files[0])
    print(f"Testing with: {image_files[0]}")

    try:
        candidates, original, processed, measurements = detector.detect_optic_disc(test_image)

        if candidates:
            best = candidates[0]
            print(f"\nDetection successful!")
            print(f"Image type       : {detector.image_type}")
            print(f"Center           : {best['center']}")
            print(f"Intensity        : {best['intensity']:.1f}")

            if detector.image_type == 'bscan':
                print(f"Nerve head height: {measurements.get('nerve_head_height', 'N/A')} px")
                print(f"Nerve head width : {measurements.get('nerve_head_width', 'N/A')} px")
            else:
                print(f"Optic disc radius: {measurements.get('optic_disc_radius', 'N/A'):.1f} px")

            onsd = measurements.get('onsd')
            area = measurements.get('posterior_globe_area')
            roc  = measurements.get('radius_of_curvature')
            print(f"ONSD             : {onsd:.1f} px" if onsd else "ONSD             : N/A")
            print(f"Post. globe area : {area:.0f} px²" if area else "Post. globe area : N/A")
            print(f"Radius of curv.  : {roc:.1f} px" if roc else "Radius of curv.  : N/A")

            if detector.image_type == 'fundus':
                crae = measurements.get('crae')
                crve = measurements.get('crve')
                avr  = measurements.get('avr')
                print(f"CRAE             : {crae:.2f} px" if crae else "CRAE             : N/A")
                print(f"CRVE             : {crve:.2f} px" if crve else "CRVE             : N/A")
                if avr:
                    norm = " (normal)" if 0.67 <= avr <= 0.75 else " (arteriolar narrowing)" if avr < 0.67 else " (wide veins)"
                    print(f"AVR              : {avr:.3f}{norm}")
                else:
                    print("AVR              : N/A")

            detector.visualize_results(candidates, original, processed, measurements)
            return best, measurements
        else:
            print("No detection found")
            return None, {}
    except Exception as e:
        print(f"Error: {e}")
        return None, {}


# ================================================================
# GUI FOLDER PICKER — open a folder and process every image in it
# ================================================================

def select_folder_and_run(pixels_per_mm=None):
    """
    Open a GUI folder-picker dialog, then run the full detection pipeline
    on every image found in the chosen folder.

    Parameters
    ----------
    pixels_per_mm : float or None
        Override the automatic scale factor (px/mm).  Pass None to use the
        built-in defaults (54 px/mm for fundus, 13 px/mm for B-scan).

    Returns
    -------
    results : list of dicts  (same format as process_all_images)
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox

    # Build a hidden root window so the dialog has a parent
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)   # bring dialog to front

    # ---- Ask for input folder ----------------------------------------
    input_dir = filedialog.askdirectory(
        title="Select folder containing fundus / B-scan images",
        mustexist=True,
    )

    if not input_dir:
        print("No folder selected — cancelled.")
        root.destroy()
        return []

    # Count images first (recursive) so the user knows what they picked
    EXTS = {'.pgm', '.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp', '.dcm'}
    image_files = []
    for _, _, files in os.walk(input_dir):
        for fname in files:
            if os.path.splitext(fname.lower())[1] in EXTS:
                image_files.append(fname)

    if not image_files:
        messagebox.showerror(
            "No images found",
            f"No supported image files found in:\n{input_dir}\n\n"
            "Supported formats: PGM, TIF, TIFF, JPG, JPEG, PNG, BMP, DCM"
        )
        root.destroy()
        return []

    # ---- Ask for output folder (default: subfolder next to input) ----
    default_out = os.path.join(input_dir, "ocular_detection_results")
    output_dir = filedialog.askdirectory(
        title=f"Select output folder  ({len(image_files)} images found — press Cancel to use default)",
        initialdir=os.path.dirname(input_dir),
    )

    if not output_dir:
        output_dir = default_out
        print(f"Using default output folder: {output_dir}")

    root.destroy()

    print(f"\nInput  folder : {input_dir}")
    print(f"Output folder : {output_dir}")
    print(f"Images found  : {len(image_files)}")
    if pixels_per_mm:
        print(f"Scale override: {pixels_per_mm} px/mm")
    else:
        print("Scale         : auto (54 px/mm fundus / 13 px/mm B-scan)")

    results = process_all_images(
        input_dir=input_dir,
        output_dir=output_dir,
        pixels_per_mm=pixels_per_mm,
    )

    # Show a short completion summary in a message box
    try:
        root2 = tk.Tk()
        root2.withdraw()
        root2.attributes('-topmost', True)
        ok   = sum(1 for r in results if r.get('status') == 'SUCCESS')
        fail = len(results) - ok
        csv_path = os.path.join(output_dir, "detection_results.csv")
        messagebox.showinfo(
            "Detection complete",
            f"Processed : {len(results)} images\n"
            f"Detected  : {ok}\n"
            f"Failed    : {fail}\n\n"
            f"CSV saved to:\n{csv_path}"
        )
        root2.destroy()
    except Exception:
        pass   # GUI summary is optional

    return results


if __name__ == "__main__":
    import sys

    # Usage:
    #   python3 improved_optic_disc_detector.py            → GUI folder picker
    #   python3 improved_optic_disc_detector.py --all <in> [<out>]
    #   python3 improved_optic_disc_detector.py <input_dir>

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        idir = sys.argv[2] if len(sys.argv) > 2 else None
        odir = sys.argv[3] if len(sys.argv) > 3 else None
        print("Processing ALL images...")
        process_all_images(input_dir=idir, output_dir=odir)
    elif len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        print("Processing SINGLE image...")
        test_detection(input_dir=sys.argv[1])
    else:
        # Default: open GUI folder picker and process everything
        select_folder_and_run()
