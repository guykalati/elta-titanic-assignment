"""Compare on development folds, freeze selection, and evaluate once on holdout."""

import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier

from data import ROOT, SEED, load_data, split_data
from features import make_preprocessor, ticket_keys
from model import fit_torch, probabilities, score

ARTIFACTS = ROOT / 'artifacts'


def candidates():
    configurations = []
    for level in ['basic', 'eda', 'extended']:
        for hidden, epochs, decay in [(0, 200, 0.03), (16, 180, 0.03), (32, 120, 0.1)]:
            configurations.append(dict(name=f'{level}_torch_{hidden}', kind='torch', level=level,
                                       hidden=hidden, epochs=epochs, lr=0.01, weight_decay=decay))
        for depth in [2, 3]:
            configurations.append(dict(name=f'{level}_xgb_{depth}', kind='xgb', level=level, depth=depth))
        configurations.append(dict(name=f'{level}_forest', kind='forest', level=level))
    return configurations


def fit_estimator(x, y, config):
    if config['kind'] == 'torch':
        return fit_torch(x, y, config)[0]
    if config['kind'] == 'xgb':
        estimator = XGBClassifier(n_estimators=250, max_depth=config['depth'], learning_rate=0.035,
                                  min_child_weight=4, subsample=0.85, colsample_bytree=0.85,
                                  reg_lambda=5, objective='binary:logistic', eval_metric='logloss',
                                  random_state=SEED, n_jobs=1)
    else:
        estimator = RandomForestClassifier(n_estimators=350, max_depth=6, min_samples_leaf=3,
                                           max_features=0.8, random_state=SEED, n_jobs=1)
    return estimator.fit(x, y)


def cross_validate(frame, config, folds):
    oof, fold_rows = np.zeros(len(frame)), []
    for fold, (train_index, test_index) in enumerate(folds, start=1):
        training, testing = frame.iloc[train_index], frame.iloc[test_index]
        processor = make_preprocessor(config['level'])
        x_train = processor.fit_transform(training)
        x_test = processor.transform(testing)
        estimator = fit_estimator(x_train, training.Survived, config)
        if config['kind'] == 'torch':
            probability = probabilities(estimator, x_test)
        else:
            probability = estimator.predict_proba(x_test)[:, 1]
        oof[test_index] = probability
        fold_rows.append(dict(fold=fold, **score(testing.Survived, probability)))
    result = score(frame.Survived, oof)
    result['fold_accuracy_std'] = float(np.std([row['accuracy'] for row in fold_rows], ddof=1))
    return result, oof, fold_rows


def travel_components(frame):
    """Keep connected ticket/family groups together for the sensitivity analysis."""
    parents = list(range(len(frame)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    size = frame.SibSp + frame.Parch + 1
    surnames = frame.Name.str.split(',', n=1).str[0].str.strip().str.lower()
    family = (surnames + '_' + size.astype(str)).where(size.gt(1))
    for keys in [ticket_keys(frame), family]:
        seen = {}
        for i, key in enumerate(keys):
            if pd.isna(key):
                continue
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    return np.array([find(i) for i in range(len(frame))])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    started = time.perf_counter()
    ARTIFACTS.mkdir(exist_ok=True)
    development, validation = split_data(load_data())
    development = development.reset_index(drop=True)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(development, development.Survived))
    configurations = candidates()
    comparison, oof_rows, fold_rows = [], [], []
    for config in configurations:
        result, oof, details = cross_validate(development, config, folds)
        comparison.append(dict(config, **result))
        oof_rows.append(pd.DataFrame({'name': config['name'], 'PassengerId': development.PassengerId,
                                     'Survived': development.Survived, 'probability': oof}))
        fold_rows.extend(dict(name=config['name'], **detail) for detail in details)
        print(f"{config['name']:<24} CV accuracy={result['accuracy']:.4f} F1={result['f1']:.4f}", flush=True)
    ranked = sorted(comparison, key=lambda row: (-row['accuracy'], row['log_loss'], row.get('hidden', 0)))
    selected_name = next(row['name'] for row in ranked if row['kind'] == 'torch')
    selected = next(config for config in configurations if config['name'] == selected_name)
    write_json(ARTIFACTS / 'selection.json', {'criterion': 'OOF accuracy, then log loss',
                                            'selected': selected, 'comparison': ranked})
    pd.concat(oof_rows, ignore_index=True).to_csv(ARTIFACTS / 'development_oof.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(ARTIFACTS / 'fold_scores.csv', index=False)

    groups = travel_components(development)
    grouped_folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(
        development, development.Survived, groups))
    grouped, grouped_probability, _ = cross_validate(development, selected, grouped_folds)
    pd.DataFrame({'PassengerId': development.PassengerId, 'group': groups,
                  'Survived': development.Survived, 'probability': grouped_probability}).to_csv(
                      ARTIFACTS / 'grouped_oof.csv', index=False)
    processor = make_preprocessor(selected['level'])
    x = processor.fit_transform(development)
    models, history = fit_torch(x, development.Survived, selected)
    joblib.dump(processor, ARTIFACTS / 'preprocessor.joblib')
    torch.save({'width': x.shape[1], 'hidden': selected['hidden'],
                'states': [model.state_dict() for model in models]}, ARTIFACTS / 'model.pt')
    manifest = {'selected': selected, 'threshold': 0.5, 'seeds': [42, 43, 44],
                'development_ids': sorted(development.PassengerId.tolist()),
                'validation_ids': sorted(validation.PassengerId.tolist()),
                'feature_count': x.shape[1], 'training_loss': history,
                'source_hashes': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ['data.py', 'features.py', 'model.py', 'train.py']},
                'holdout_used_for_selection': False}
    write_json(ARTIFACTS / 'manifest.json', manifest)
    # Selection and weights are saved before the final validation labels are evaluated.
    probability = probabilities(models, processor.transform(validation))
    holdout = score(validation.Survived, probability)
    evaluated = validation.assign(probability=probability, prediction=(probability >= 0.5).astype(int))
    evaluated.to_csv(ARTIFACTS / 'validation_predictions.csv', index=False)
    validation.to_csv(ROOT / 'data' / 'validation.csv', index=False)
    validation.drop(columns='Survived').head(8).to_csv(ROOT / 'data' / 'inference_sample.csv', index=False)
    baseline = score(validation.Survived, np.full(len(validation), development.Survived.mean()))
    write_json(ARTIFACTS / 'metrics.json', {'holdout': holdout, 'majority_holdout': baseline,
                                          'grouped_cv': grouped, 'development_cv': ranked,
                                          'group_count': int(len(np.unique(groups))),
                                          'elapsed_seconds': time.perf_counter() - started})
    subgroup_rows = []
    for variable in ['Sex', 'Pclass']:
        for value, subset in evaluated.groupby(variable):
            subgroup_rows.append(dict(variable=variable, value=str(value), **score(subset.Survived, subset.probability)))
    pd.DataFrame(subgroup_rows).to_csv(ARTIFACTS / 'validation_subgroups.csv', index=False)
    print(f"Selected {selected_name}; holdout accuracy={holdout['accuracy']:.4f}; grouped CV={grouped['accuracy']:.4f}")


if __name__ == '__main__':
    main()
