import unittest

import pandas as pd

from modeling.prepare_modeling_dataset_v2 import prepare_dataset


class PrepareModelingDatasetV2Tests(unittest.TestCase):
    def test_adds_october_baseline_filters_small_groups_and_calculates_y(self):
        source = pd.DataFrame(
            [
                ("20251001", "Region A", "Industry A", 100.0),
                ("20251002", "Region A", "Industry A", 200.0),
                ("20250701", "Region A", "Industry A", 300.0),
                ("20251001", "Region B", "Industry B", 50.0),
                ("20250701", "Region B", "Industry B", 75.0),
            ],
            columns=["TA_YMD", "MCT_SGG_CD", "MCT_RY_CD", "TS_AT"],
        )

        result = prepare_dataset(source, minimum_october_days=2)

        self.assertEqual(len(result), 3)
        self.assertEqual(set(result["MCT_SGG_CD"]), {"Region A"})
        self.assertTrue(result["october_n"].eq(2).all())
        self.assertTrue(result["october_mean"].eq(150.0).all())
        self.assertEqual(result.loc[result["TA_YMD"].eq("20250701"), "Y"].iloc[0], 100.0)
        self.assertEqual(
            list(result.columns),
            [*source.columns, "october_mean", "october_n", "Y"],
        )

    def test_requires_october_baselines(self):
        source = pd.DataFrame(
            [("20250701", "Region A", "Industry A", 100.0)],
            columns=["TA_YMD", "MCT_SGG_CD", "MCT_RY_CD", "TS_AT"],
        )

        with self.assertRaisesRegex(ValueError, "no October rows"):
            prepare_dataset(source)

    def test_rejects_duplicate_date_region_industry_keys(self):
        source = pd.DataFrame(
            [
                ("20251001", "Region A", "Industry A", 100.0),
                ("20251001", "Region A", "Industry A", 200.0),
            ],
            columns=["TA_YMD", "MCT_SGG_CD", "MCT_RY_CD", "TS_AT"],
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare_dataset(source)


if __name__ == "__main__":
    unittest.main()
