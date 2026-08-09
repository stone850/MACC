import unittest

import yaml

from collect_offline_dataset import _runtime_versions


class CollectionMetadataTest(unittest.TestCase):
    def test_runtime_versions_are_yaml_safe(self):
        versions = _runtime_versions()
        self.assertEqual(set(versions), {
            "python",
            "torch",
            "numpy",
            "gym",
            "lbforaging",
            "sacred",
        })
        for value in versions.values():
            self.assertTrue(value is None or type(value) is str)
        yaml.safe_dump({"runtime_versions": versions})


if __name__ == "__main__":
    unittest.main()
