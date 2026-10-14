"""Training-only preprocessing and row-local Titanic features."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED = {'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'}


def check_inputs(frame):
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError('The CSV contains no passengers.')
    frame = frame.copy()
    for column in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
        values = pd.to_numeric(frame[column], errors='coerce')
        if (frame[column].notna() & values.isna()).any() or np.isinf(values).any():
            raise ValueError(f'{column} must contain finite numbers or empty values.')
        if values.dropna().lt(0).any():
            raise ValueError(f'{column} cannot be negative.')
        frame[column] = values.astype(float)
    if not frame.Pclass.isin([1, 2, 3]).all():
        raise ValueError('Pclass must be 1, 2 or 3.')
    for column in ['SibSp', 'Parch']:
        if frame[column].isna().any() or frame[column].mod(1).ne(0).any():
            raise ValueError(f'{column} must contain nonnegative whole numbers.')
    if not frame.Sex.isin(['male', 'female']).all():
        raise ValueError('Sex must be male or female.')
    if not frame.Embarked.dropna().isin(['C', 'Q', 'S']).all():
        raise ValueError('Embarked must be C, Q, S or empty.')
    return frame


def titles(frame):
    names = frame.Name.astype('string')
    return names.str.extract(r',\s*([^.]*)\.', expand=False).str.strip().fillna('Unknown')


def ticket_keys(frame):
    return frame.Ticket.astype('string').str.upper().str.replace(r'\s+', '', regex=True)


class TitanicFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, level='basic'):
        self.level = level

    def fit(self, frame, y=None):
        frame = check_inputs(frame)
        self.age_median_ = float(frame.Age.median())
        self.group_age_ = frame.assign(Title=titles(frame)).groupby(['Title', 'Pclass']).Age.median().to_dict()
        self.ticket_counts_ = ticket_keys(frame).value_counts().to_dict()
        return self

    def transform(self, frame):
        frame = check_inputs(frame)
        result = frame[['Age', 'Fare', 'SibSp', 'Parch', 'Sex', 'Pclass', 'Embarked']].copy()
        result['Pclass'] = frame.Pclass.astype(int).astype(str)
        result['Embarked'] = frame.Embarked.fillna('Unknown')
        if self.level != 'basic':
            raw_title = titles(frame)
            title = raw_title.replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
            result['Title'] = title.where(title.isin(['Mr', 'Mrs', 'Miss', 'Master']), 'Other')
            result['AgeMissing'] = frame.Age.isna().astype(float)
            result['CabinKnown'] = frame.Cabin.notna().astype(float)
            result['FamilySize'] = frame.SibSp + frame.Parch + 1
            family_group = pd.cut(result.FamilySize, [0, 1, 4, np.inf], labels=['Alone', 'Small', 'Large'])
            result['FamilyGroup'] = family_group.astype(str)
            result['SexClass'] = frame.Sex + '_' + result.Pclass
            result['Fare'] = np.log1p(frame.Fare)
        if self.level == 'extended':
            estimates = [
                self.group_age_.get((title, pclass), self.age_median_)
                for title, pclass in zip(raw_title, frame.Pclass)
            ]
            result['Age'] = frame.Age.fillna(pd.Series(estimates, index=frame.index)).fillna(self.age_median_)
            child = pd.Series(np.where(result.Age.lt(16), 'Child', 'Adult'), index=frame.index)
            result['SexLifeStageClass'] = result.SexClass + '_' + child
            result['SexFamily'] = frame.Sex + '_' + result.FamilyGroup
            result['Deck'] = frame.Cabin.astype('string').str.extract(r'([A-Za-z])', expand=False).str.upper().fillna('Unknown')
            result['CabinCount'] = frame.Cabin.fillna('').str.split().str.len().astype(float)
            ticket_count = ticket_keys(frame).map(self.ticket_counts_).fillna(1).astype(float)
            result['ObservedTicketCount'] = ticket_count
            result['LogFarePerObservedMember'] = np.log1p(frame.Fare / ticket_count)
            prefix = frame.Ticket.astype('string').str.upper().str.replace(r'[^A-Z]', '', regex=True)
            result['TicketPrefix'] = prefix.replace('', 'Numeric').fillna('Unknown')
        return result


def make_preprocessor(level):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical = OneHotEncoder(handle_unknown='ignore', min_frequency=3, sparse_output=False)
    columns = ColumnTransformer([
        ('numeric', numeric, make_column_selector(dtype_include=np.number)),
        ('category', categorical, make_column_selector(dtype_exclude=np.number)),
    ])
    return Pipeline([('features', TitanicFeatures(level)), ('columns', columns)])
