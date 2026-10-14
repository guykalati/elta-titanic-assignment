"""Post-selection development diagnostics; never update selection or final weights."""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer

from data import ROOT, SEED, load_data
from features import make_preprocessor
from model import fit_torch, probabilities, score

BLOCKS = {
    'without_title': ['Title'],
    'without_family': ['FamilySize', 'FamilyGroup', 'SexFamily'],
    'without_demographic_interactions': ['SexClass', 'SexLifeStageClass'],
    'without_cabin': ['CabinKnown', 'Deck', 'CabinCount'],
    'without_ticket': ['ObservedTicketCount', 'LogFarePerObservedMember', 'TicketPrefix'],
}


def alter_features(frame, name):
    frame = frame.copy()
    if name == 'global_numeric_age':
        frame.loc[frame.AgeMissing.eq(1), 'Age'] = np.nan
    if name == 'raw_numeric_fare':
        frame['Fare'] = np.expm1(frame.Fare)
    return frame.drop(columns=BLOCKS.get(name, []))


def main():
    artifacts = ROOT / 'artifacts'
    manifest = json.loads((artifacts / 'manifest.json').read_text())
    config = manifest['selected']
    assert config['level'] == 'extended', 'These diagnostics describe the extended feature set.'
    reference = pd.read_csv(artifacts / 'development_oof.csv')
    reference = reference.loc[reference.name.eq(config['name'])].reset_index(drop=True)
    assert reference.PassengerId.is_unique
    assert set(reference.PassengerId) == set(manifest['development_ids'])
    frame = load_data().set_index('PassengerId', drop=False).loc[reference.PassengerId].reset_index(drop=True)
    np.testing.assert_array_equal(frame.Survived, reference.Survived)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(frame, frame.Survived))
    results = []
    for name in ['baseline', *BLOCKS, 'global_numeric_age', 'raw_numeric_fare']:
        oof = np.zeros(len(frame))
        for train_index, test_index in folds:
            training, testing = frame.iloc[train_index], frame.iloc[test_index]
            processor = make_preprocessor(config['level'])
            processor.steps.insert(1, ('diagnostic', FunctionTransformer(alter_features, kw_args={'name': name})))
            x_train = processor.fit_transform(training)
            x_test = processor.transform(testing)
            models, _ = fit_torch(x_train, training.Survived, config)
            oof[test_index] = probabilities(models, x_test)
        if name == 'baseline':
            np.testing.assert_allclose(oof, reference.probability, rtol=1e-6, atol=1e-7)
        metrics = score(frame.Survived, oof)
        results.append({'name': name, 'accuracy': metrics['accuracy'],
                        'f1': metrics['f1'], 'roc_auc': metrics['roc_auc']})
        print(f"{name:<34} development OOF accuracy={metrics['accuracy']:.4f}", flush=True)
    table = pd.DataFrame(results)
    table['delta_accuracy_vs_baseline'] = table.accuracy - table.accuracy.iloc[0]
    table.to_csv(artifacts / 'ablation.csv', index=False)


if __name__ == '__main__':
    main()
