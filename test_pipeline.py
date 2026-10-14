"""Checks for preprocessing boundaries, inference consistency, and saved results."""

import csv
import json
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from data import ROOT
from features import check_inputs, make_preprocessor
from model import load_bundle, predict
from train import travel_components


def sample():
    return pd.DataFrame({
        'PassengerId': [1, 2, 3, 4, 5, 6], 'Survived': [0, 1, 0, 1, 0, 1],
        'Pclass': [1, 1, 2, 2, 3, 3],
        'Name': ['Smith, Mr. A', 'Smith, Mrs. B', 'Jones, Mr. C',
                 'Jones, Master. D', 'Brown, Miss. E', 'Hill, Mr. F'],
        'Sex': ['male', 'female', 'male', 'male', 'female', 'male'],
        'Age': [20., 30., 40., 10., 25., np.nan],
        'SibSp': [1, 1, 0, 1, 0, 0], 'Parch': [0, 0, 1, 0, 0, 0],
        'Ticket': ['A 1', 'A 1', 'B 2', 'B 2', '3', '4'],
        'Fare': [60., 60., 20., 20., 8., 7.],
        'Cabin': ['C1', 'C1', None, None, None, None],
        'Embarked': ['S', 'S', 'C', 'C', 'Q', 'Q'],
    })


class PipelineChecks(unittest.TestCase):
    def test_target_does_not_enter_features(self):
        frame = sample()
        changed = frame.assign(Survived=1 - frame.Survived)
        for level in ['basic', 'eda', 'extended']:
            with self.subTest(level=level):
                first, second = make_preprocessor(level), make_preprocessor(level)
                np.testing.assert_array_equal(first.fit_transform(frame), second.fit_transform(changed))
                np.testing.assert_array_equal(first.transform(frame), first.transform(frame.drop(columns='Survived')))

    def test_training_statistics_and_batch_invariance(self):
        frame = sample()
        processor = make_preprocessor('extended').fit(frame)
        features = processor.named_steps['features']
        self.assertEqual(features.age_median_, 25.)
        self.assertEqual(features.group_age_[('Mr', 1.)], 20.)
        self.assertEqual(features.ticket_counts_['A1'], 2)
        original_counts = features.ticket_counts_.copy()
        original_groups = features.group_age_.copy()
        unseen = frame.iloc[[0]].assign(Ticket='NEW PREFIX 999', Cabin='Z99',
                                        Name='New, Duke. A', Age=np.nan, Fare=np.nan)
        batch = pd.concat([frame, unseen], ignore_index=True)
        transformed = processor.transform(batch)
        self.assertTrue(np.isfinite(transformed).all())
        for index in range(len(batch)):
            np.testing.assert_allclose(transformed[index], processor.transform(batch.iloc[[index]])[0])
        permutation = [6, 3, 0, 5, 2, 1, 4]
        np.testing.assert_allclose(processor.transform(batch.iloc[permutation]), transformed[permutation])
        processor.transform(frame.assign(Age=1000., Ticket='NEW PREFIX 999'))
        self.assertEqual(features.ticket_counts_, original_counts)
        self.assertEqual(features.group_age_, original_groups)
        self.assertEqual(features.age_median_, 25.)
        numeric = processor.named_steps['columns'].named_transformers_['numeric']
        age_position = processor.named_steps['columns'].transformers_[0][2].index('Age')
        self.assertEqual(numeric.named_steps['impute'].statistics_[age_position], 25.)

    def test_travel_components_keep_transitive_links_together(self):
        frame = sample()
        # Rows 0/1 share a ticket; rows 1/2 share family; rows 2/3 share a ticket.
        frame.loc[2, 'Name'] = 'Smith, Mr. C'
        groups = travel_components(frame)
        self.assertEqual(len(set(groups[:4])), 1)
        folds = StratifiedGroupKFold(2, shuffle=True, random_state=42).split(frame, frame.Survived, groups)
        for training, testing in folds:
            self.assertFalse(set(groups[training]) & set(groups[testing]))

    def test_invalid_inputs_are_rejected(self):
        frame = sample()
        for column in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
            for invalid in ['bad number', -1, np.inf]:
                with self.subTest(column=column, invalid=invalid), self.assertRaises(ValueError):
                    check_inputs(frame.assign(**{column: invalid}))
        for column, invalid in [('Pclass', 4), ('SibSp', 0.5), ('Parch', np.nan),
                                ('Sex', 'unknown'), ('Embarked', 'X')]:
            with self.subTest(column=column, invalid=invalid), self.assertRaises(ValueError):
                check_inputs(frame.assign(**{column: invalid}))
        for missing in ['Age', 'Ticket', 'Name']:
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                check_inputs(frame.drop(columns=missing))
        with self.assertRaises(ValueError):
            check_inputs(frame.iloc[:0])

    def test_saved_predictions_and_independent_metrics(self):
        artifact = ROOT / 'artifacts'
        if not (artifact / 'metrics.json').exists():
            self.skipTest('Training artifacts have not been generated yet.')
        frame = pd.read_csv(artifact / 'validation_predictions.csv')
        bundle = load_bundle(artifact)
        probability = predict(frame, bundle)
        np.testing.assert_allclose(probability, frame.probability, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(predict(frame.iloc[::-1], bundle), probability[::-1], rtol=1e-6, atol=1e-7)
        for position in [0, len(frame) // 2, len(frame) - 1]:
            np.testing.assert_allclose(predict(frame.iloc[[position]], bundle), probability[[position]], rtol=1e-6, atol=1e-7)
        with (artifact / 'validation_predictions.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        matrix = [[0, 0], [0, 0]]
        for row in rows:
            truth = int(row['Survived'])
            predicted = int(float(row['probability']) >= 0.5)
            self.assertEqual(predicted, int(row['prediction']))
            matrix[truth][predicted] += 1
        metrics = json.loads((artifact / 'metrics.json').read_text())['holdout']
        self.assertEqual(matrix, metrics['confusion_matrix'])
        self.assertAlmostEqual((matrix[0][0] + matrix[1][1]) / len(rows), metrics['accuracy'])


if __name__ == '__main__':
    unittest.main()
