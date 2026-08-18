"""Unit test suite for Finding Lanes computer vision pipeline."""

import os
import unittest
import numpy as np
import cv2

# Support running directly or as package
try:
    from lanes import (
        canny,
        roi,
        make_coordinate,
        average_lines_intercept,
        display_lines,
        process_frame,
        resolve_asset_path,
    )
except ImportError:
    from Finding_Lanes.lanes import (
        canny,
        roi,
        make_coordinate,
        average_lines_intercept,
        display_lines,
        process_frame,
        resolve_asset_path,
    )


class TestFindingLanes(unittest.TestCase):
    """Test suite verifying all core functions of the Finding Lanes pipeline."""

    def setUp(self):
        """Create test image frames of various dimensions."""
        self.height, self.width = 720, 1280
        # Create a synthetic 3-channel BGR image
        self.test_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Draw synthetic left lane (negative slope in image coordinates)
        cv2.line(self.test_frame, (300, 720), (580, 450), (255, 255, 255), 8)
        # Draw synthetic right lane (positive slope in image coordinates)
        cv2.line(self.test_frame, (1000, 720), (700, 450), (255, 255, 255), 8)

    def test_canny_edge_detection(self):
        """Verify Canny edge detector outputs a binary 2D edge map."""
        edges = canny(self.test_frame)
        self.assertEqual(edges.shape, (self.height, self.width))
        self.assertEqual(edges.dtype, np.uint8)
        self.assertTrue(np.any(edges > 0), "Canny edge detector should find drawn lines")

    def test_canny_grayscale_input(self):
        """Verify Canny handles single-channel 2D grayscale input gracefully."""
        gray = cv2.cvtColor(self.test_frame, cv2.COLOR_BGR2GRAY)
        edges = canny(gray)
        self.assertEqual(edges.shape, (self.height, self.width))

    def test_dynamic_roi_masking(self):
        """Verify dynamic ROI preserves road area and masks out non-ROI regions."""
        edges = canny(self.test_frame)
        masked = roi(edges)
        self.assertEqual(masked.shape, edges.shape)
        # Top corners should be masked out (all zeros)
        self.assertEqual(masked[0, 0], 0)
        self.assertEqual(masked[0, self.width - 1], 0)

    def test_roi_custom_polygon(self):
        """Verify ROI accepts custom polygon coordinates."""
        custom_poly = np.array([[(100, 700), (800, 700), (450, 300)]], dtype=np.int32)
        edges = canny(self.test_frame)
        masked = roi(edges, polygons=custom_poly)
        self.assertEqual(masked.shape, edges.shape)

    def test_make_coordinate_valid(self):
        """Verify make_coordinate computes expected pixel coordinates from slope & intercept."""
        # Slope = -1.0, Intercept = 1000
        coords = make_coordinate(self.test_frame, (-1.0, 1000.0))
        self.assertIsNotNone(coords)
        self.assertEqual(len(coords), 4)
        x1, y1, x2, y2 = coords
        self.assertEqual(y1, self.height)
        self.assertEqual(y2, int(self.height * 0.6))
        self.assertEqual(x1, int((720 - 1000) / -1.0))

    def test_make_coordinate_edge_cases(self):
        """Verify make_coordinate handles zero slopes, NaNs, infinities, and None inputs."""
        self.assertIsNone(make_coordinate(self.test_frame, None))
        self.assertIsNone(make_coordinate(self.test_frame, []))
        self.assertIsNone(make_coordinate(self.test_frame, (0.0, 500.0)))
        self.assertIsNone(make_coordinate(self.test_frame, (np.nan, 500.0)))
        self.assertIsNone(make_coordinate(self.test_frame, (-1.0, np.inf)))

    def test_average_lines_intercept_none_and_empty(self):
        """Verify average_lines_intercept handles None and empty line inputs."""
        self.assertIsNone(average_lines_intercept(self.test_frame, None))
        self.assertIsNone(average_lines_intercept(self.test_frame, np.array([])))

    def test_average_lines_slope_filtering(self):
        """Verify horizontal noise lines (near zero slope) are filtered out."""
        # Horizontal line: (100, 500) to (900, 500) -> slope = 0
        horizontal_line = np.array([[[100, 500, 900, 500]]])
        result = average_lines_intercept(self.test_frame, horizontal_line, min_slope=0.3)
        self.assertIsNone(result)

    def test_average_lines_detection(self):
        """Verify left and right lanes are correctly categorized and averaged."""
        left_seg = [[300, 720, 580, 450]]
        right_seg = [[1000, 720, 700, 450]]
        lines = np.array([left_seg, right_seg])
        lane_lines = average_lines_intercept(self.test_frame, lines)
        self.assertIsNotNone(lane_lines)
        self.assertEqual(len(lane_lines), 2)

    def test_display_lines(self):
        """Verify display_lines generates an overlay image with the correct shape."""
        lanes = np.array([[300, 720, 580, 432], [1000, 720, 700, 432]])
        line_img = display_lines(self.test_frame, lanes)
        self.assertEqual(line_img.shape, self.test_frame.shape)
        self.assertEqual(line_img.dtype, np.uint8)

    def test_display_lines_none(self):
        """Verify display_lines returns all-black canvas when lines is None."""
        line_img = display_lines(self.test_frame, None)
        self.assertEqual(line_img.shape, self.test_frame.shape)
        self.assertTrue(np.all(line_img == 0))

    def test_process_frame_end_to_end(self):
        """Verify full process_frame pipeline runs successfully on synthetic frame."""
        combo_image, lanes = process_frame(self.test_frame)
        self.assertEqual(combo_image.shape, self.test_frame.shape)
        self.assertIsNotNone(lanes)

    def test_process_frame_intermediates(self):
        """Verify process_frame returns intermediate dictionary when requested."""
        combo, lanes, intermediates = process_frame(self.test_frame, return_intermediates=True)
        self.assertIn("canny", intermediates)
        self.assertIn("roi", intermediates)
        self.assertIn("line_image", intermediates)
        self.assertIn("raw_lines", intermediates)

    def test_process_bundled_image_if_present(self):
        """Verify pipeline execution on bundled picture.jpg asset."""
        img_path = resolve_asset_path("picture.jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            self.assertIsNotNone(img)
            combo, lanes = process_frame(img)
            self.assertEqual(combo.shape, img.shape)
            self.assertIsNotNone(lanes)
            self.assertEqual(len(lanes), 2, "Should detect both left and right lane lines")


if __name__ == "__main__":
    unittest.main()
