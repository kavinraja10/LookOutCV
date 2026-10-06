import unittest

import look_out_cv


class TestLookOutCV(unittest.TestCase):
    def test_imports(self):
        """Test that key components are accessible from the top-level package."""
        expected_components = [
            "ClassificationLogger",
            "DataCollectionLogger",
            "DetectionLogger",
            "CVMetrics",
            "MetricsRollup",
            "DataRetentionManager",
            "RetentionPolicy",
            "DriftDetector",
            "DriftTest",
            "DriftReport",
        ]

        for component in expected_components:
            self.assertTrue(
                hasattr(look_out_cv, component),
                f"look_out_cv is missing {component}",
            )

    def test_all_declaration(self):
        """Test that __all__ matches the expected components."""
        expected_components = [
            "ClassificationLogger",
            "DataCollectionLogger",
            "DetectionLogger",
            "CVMetrics",
            "MetricsRollup",
            "DataRetentionManager",
            "RetentionPolicy",
            "DriftDetector",
            "DriftTest",
            "DriftReport",
        ]
        
        self.assertCountEqual(
            look_out_cv.__all__,
            expected_components,
            "__all__ declaration does not match expected components",
        )

if __name__ == "__main__":
    unittest.main()
