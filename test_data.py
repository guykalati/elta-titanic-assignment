"""Small offline contract checks; artificial records are not analysis evidence."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd

import data


class DataChecks(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "PassengerId": range(1, 31), "Survived": [0, 1] * 15,
            "Pclass": [1, 2, 3] * 10, "Name": ["Fixture, Mr. Example"] * 30,
            "Sex": ["male", "female"] * 15, "Age": [20.] * 30,
            "SibSp": [0] * 30, "Parch": [0] * 30, "Ticket": ["fixture"] * 30,
            "Fare": [10.] * 30, "Cabin": [None] * 30, "Embarked": ["S"] * 30,
        })

    def test_split_is_repeatable_disjoint_and_guarded(self):
        with TemporaryDirectory() as directory:
            with patch.object(data, "SPLIT_PATH", Path(directory) / "split.json"):
                development, validation = data.split_data(self.frame)
                repeated, _ = data.split_data(self.frame)
                self.assertTrue(development.equals(repeated))
                self.assertEqual((len(development), len(validation)), (24, 6))
                self.assertTrue(set(development.PassengerId).isdisjoint(validation.PassengerId))
                changed = self.frame.copy()
                changed.loc[0, "Fare"] = 11
                with self.assertRaises(ValueError):
                    data.split_data(changed)

    def test_rejects_wrong_schema_labels_and_duplicate_ids(self):
        for invalid in (self.frame.drop(columns="Age"), self.frame.assign(Survived=2),
                        self.frame.assign(PassengerId=1)):
            with self.subTest():
                with self.assertRaises(ValueError):
                    data.validate_data(invalid)

    def test_download_requests_and_extracts_only_train(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"

            def fake_download(command, **kwargs):
                self.assertEqual(command[command.index("-f") + 1], "train.csv")
                with ZipFile(path.with_suffix(".csv.zip"), "w") as archive:
                    archive.writestr("train.csv", self.frame.to_csv(index=False))
                    archive.writestr("../unwanted.txt", "never extract arbitrary members")
                return SimpleNamespace(returncode=0)

            with patch.object(data, "DATA_PATH", path), patch.object(data.subprocess, "run", fake_download):
                data.download_train()
                self.assertEqual(len(pd.read_csv(path)), 30)
                self.assertEqual([item.name for item in path.parent.iterdir()], ["train.csv"])


if __name__ == "__main__":
    unittest.main()
