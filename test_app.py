"""Run after training: python -m unittest test_app.py."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent


class AppChecks(unittest.TestCase):
    def test_report_and_csv_inputs(self):
        validation = pd.read_csv(ROOT / "artifacts" / "validation_predictions.csv")
        app = AppTest.from_file(str(ROOT / "ds_app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.metric), 4)
        self.assertEqual(app.dataframe[0].value.shape[1], 7)

        cases = {
            "labeled": (validation, None),
            "label_free": (validation.drop(columns="Survived"), None),
            "single_class": (validation.loc[validation.Survived.eq(0)], None),
            "missing_column": (validation.drop(columns="Age"), "Missing columns: Age"),
            "invalid_number": (validation.assign(Age="bad"), "Age must contain finite numbers"),
            "invalid_labels": (validation.assign(Survived=2), "Survived must contain only 0 or 1"),
            "empty_rows": (validation.iloc[:0], "The CSV has no passenger rows."),
        }
        with TemporaryDirectory(prefix="elta-app-check-") as directory:
            for name, (frame, expected_error) in cases.items():
                with self.subTest(input=name):
                    path = Path(directory) / f"{name}.csv"
                    frame.to_csv(path, index=False)
                    app.text_input[0].set_value(str(path))
                    app.button[0].click().run()
                    self.assertEqual(len(app.exception), 0)
                    if expected_error:
                        self.assertEqual(len(app.error), 1)
                        self.assertIn(expected_error, app.error[0].value)
                        self.assertEqual(len(app.dataframe), 1)
                    else:
                        self.assertEqual(len(app.error), 0)
                        result = app.dataframe[-1].value
                        self.assertEqual(len(result), len(frame))
                        self.assertTrue(np.isfinite(result.probability).all())
                        self.assertTrue(result.probability.between(0, 1).all())
                        self.assertTrue(result.prediction.isin([0, 1]).all())
                        if name == "label_free":
                            self.assertEqual(len(app.metric), 4)
                        if name == "single_class":
                            self.assertTrue(any("ROC AUC unavailable" in item.value
                                                for item in app.caption))
            app.text_input[0].set_value(str(Path(directory) / "absent.csv"))
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("CSV not found", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
