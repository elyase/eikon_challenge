# -*- coding: utf-8 -*-
from __future__ import division
from utils import *
import json
from sklearn.base import TransformerMixin
import inspect
import jellyfish as jf
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import StandardScaler
import numpy as np
from candidates import is_tagged_candidate, get_relevance
from sklearn.preprocessing import balance_weights


class MentionFeatures(TransformerMixin):

    """Mention related features. Returns a dict if used on
    a single mention and a pd.DataFrame if used on a story or stories.
    Example:
        """

    def __init__(self, kb, stories, mention=None):
        self.kb = kb
        self.stories = stories
        if mention is not None:
            self.mention = mention
            self.extract_features()

    def extract_features(self):
        self.features = {}
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith('mention_'):
                self.features.update({name: method()})

        extracted_features = {k: v for k, v in self.mention.iteritems()}
        self.features.update(extracted_features)
        return self

    def __repr__(self):
        return (self.features['text'] + ' '
                + json.dumps(self.features, sort_keys=True, indent=2))

    #**********************************************************#
    #                Features section
    #**********************************************************#
    def mention_is_ticker(self):
        return self.mention['text'].upper() in self.kb.tickers

    def mention_is_one_word(self):
        """Could be a NIL indicator. Ex: "Purchased"
        """
        return self.mention['token_cnt'] == 1

    def mention_character_length(self):
        return len(self.mention['text'])

    def mention_follows_copyright(self):
        """Ex: "© 2012 AT&amp;T Intellectual Property.
               "©2012 Teleflex Incorporated"
        """
        regex = u"©\s?[1|2]\d{3}"
        s = self.mention['previous_word'] + self.mention['previous_word2']
        return True if re.match(regex, s) else False

    def mention_prev_is_about(self):
        """Ex: "About Walmart"
        """
        return self.mention['previous_word'] == u'About'

    def mention_prev2_is_SYMBOL(self):
        """Ex: "SYMBOL: RBPAA (CL A)"
        """
        return self.mention['previous_word2'] == u'SYMBOL'

    def mention_prev2_is_ISSUER(self):
        """Ex: "ISSUER: ROYAL BANCSHARES OF PENN INC"
        """
        return self.mention['previous_word2'] == u'ISSUER'

    def mention_followed_by_8K(self):
        """Ex: "Graymark Healthcare, Inc. 8-K"
        """
        is_headline = self.mention['paragraph'] == 0
        contains_8K = '8-K' in self.mention['next_word'] + \
            self.mention['next_word2']
        return is_headline and contains_8K

    def mention_followed_by_logo(self):
        """Ex: "The New World Systems logo is available ..."
        """
        s = self.mention['next_word'] + self.mention['next_word2']
        return 'logo' in s

    def mention_followed_by_orthe(self):
        """Ex: Parkland Fuel Corporation ("Parkland" or the "Corporation")"""
        s = self.mention['next_word'] + self.mention['next_word2']
        return s == 'orthe'

    def mention_token_score(self):
        """Add token scores. The score is inversely proportional to the token
        frequency in training data and proportional to the frequency in the
        KnowledgBase. For example
            token_score("of")->low
            token_score("Inc")->high"""
        acc = 0
        tokens = normalized(self.mention['text'])
        for token in tokens:
            company_freq = self.kb.company_tokens_freqs.get(token, 0)
            document_freq = self.stories.token_freqs.get(token, 1)
            acc += company_freq / document_freq
        return acc / len(tokens)

    def mention_is_exact_match(self):
        return self.mention['text'] in self.kb.full_names

    def mention_is_lowercase_match(self):
        return self.mention['text'].lower() in self.kb.full_names_lower

    def mention_is_surface_form(self):
        return self.mention['text'] in self.kb.surface_forms

    def mention_is_token_match(self):
        """All lowercase tokens match"""
        return normalized(self.mention['text'], tuples=True) in self.kb.normalized_companies

    #**********************************************************#
    #    Required for use in scikits Pipeline, FeatureUnion
    #**********************************************************#
    def transform(self, mentions, **transform_params):
        if isinstance(mentions, pd.DataFrame):
            mentions = mentions.to_dict(outtype='records')

        all_features = []
        for mention in mentions:
            _feats = MentionFeatures(self.kb, self.stories, mention)
            all_features.append(_feats.features)

        df = pd.DataFrame(all_features)
        #**********************************************************#
        #              Document aggregate features
        #**********************************************************#
        # Calculate mention counts per document
        df['mention_count'] = df.groupby(
            ['story_id', 'text'])['text'].transform('count')

        # normalize counts per document
        def normalize_counts(grp):
            grp['mention_count'] = grp[
                'mention_count'] / grp['mention_count'].sum()
            return grp
        df = df.groupby('story_id').apply(normalize_counts)
        return df  # df.to_dict(outtype='records')

    def fit(self, X, y=None, **fit_params):
        return self


class DocumentFeatures(TransformerMixin):

    """Document related features with the purpose of determining wether
    there should be tags at all, or find out its relevance.
    From the challenge description:
        > "Relevance tags should be listed as high (‘H’), medium (‘M’), or low (‘L’)
        > by the Solver. In addition, many news items are not specific to a particular
        > company or organization and therefore should not be tagged at all"
    """
    doc_cache = defaultdict(list)

    def __init__(self, kb, stories, doc=None):
        self.kb = kb
        self.stories = stories
        if doc is not None:
            self.doc = doc
            if doc['story_id'] in self.doc_cache:
                self.doc_token_cnt = self.doc_cache[doc['story_id']]
            else:
                self.doc_token_cnt = len(normalized(doc['headline'] + doc['text'],
                                                    fix=False))
                self.doc_cache[doc['story_id']] = self.doc_token_cnt
            self.extract_features()

    def extract_features(self):
        """Runs all methods starting with _doc and updates self.features
        """
        self.features = {}
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith('doc_'):
                self.features.update({name: method()})
        return self

    def __repr__(self):
        return (self.doc['headline'][:35] + '...'
                + json.dumps(self.features, sort_keys=True, indent=2))

    #**********************************************************#
    #                Features section
    #**********************************************************#
    # number of digits ratio, contains_About, 8k in headline
    # def ticker_ratio(self):
    #    pass
    #    return self.mention['text'].upper() in kb.tickers
    def doc_token_cnt(self):
        return self.doc_token_cnt

    def doc_character_length(self):
        return len(self.doc['text'])

    def doc_number_count(self):
        """Find the count of all floating like numbers"""
        return len(re.findall(r"\d+[\.|,]\d+", self.doc['text']))

    def doc_number_ratio(self):
        """Find the ratio of all floating like numbers"""
        return self.doc_number_count() / self.doc_token_cnt

    def doc_contains_form4(self):
        """Indicative of high relevance (insider trading)
        """
        return 'Form 4' in self.doc['text']

    def doc_DJ_headline(self):
        """Indicative of no company tags
        """
        result = 0 if self.doc['headline'].startswith('DJ') else 1
        if 'Form 4' in self.doc['text'] or '8-K' in self.doc['headline']:
            result = 1
        return result

    def doc_source(self):
        return self.doc['source']

    def doc_countries(self):
        text = self.doc['text'] + self.doc['text']
        return text_countries(text, COUNTRY_CODES)

    # NOTE: Already in Mentions
    # def doc_contains_about(self):
    #    """Ex: "About Walmart"
    #    """
    #    return 'About' in self.doc['text']
    #**********************************************************#
    #    Required for use in scikits Pipeline, FeatureUnion
    #**********************************************************#
    def transform(self, mentions, **transform_params):
        if isinstance(mentions, pd.DataFrame):
            mentions = mentions.to_dict(outtype='records')

        all_features = []
        for mention in mentions:
            doc = self.stories.by_id(mention['story_id'])
            _feats = DocumentFeatures(self.kb, self.stories, doc)
            mention.update(_feats.features)
            all_features.append(mention)

        df = pd.DataFrame(all_features)
        #**********************************************************#
        #              Document aggregate features
        #**********************************************************#
        # Calculate mention counts per document
        return df  # df.to_dict(outtype='records')

    def fit(self, X, y=None, **fit_params):
        return self


class MentionCandidateFeatures(TransformerMixin):

    """Mention candidates Features
    In this class `mc` is short for mention_candidate
    """

    def __init__(self, kb, stories, mc=None, append_features=True):
        """
        Parameters:
            append_features: wether to append mentions features or just
            output the mc features
        """
        self.kb = kb
        self.stories = stories
        self.append_features = append_features

        if mc is not None:
            self.mc = mc
            self.mention_text = self.mc['text']
            self.candidate_name = self.kb.companies.name.get(
                self.mc['candidate'])
            self.extract_features()

    def extract_features(self):
        """Runs all methods starting with _doc and updates self.features
        """
        self.features = {}
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith('mc_'):
                self.features.update({name: method()})
        return self

    # def __repr__(self):
    #    return (self.mc['text'] + '...'
    #            + json.dumps(self.features, sort_keys=True, indent=2))

    #**********************************************************#
    #                Features section
    #**********************************************************#
    # def mc_common_tokens(self):
    #    """The count of the common tokens between the Mention and the Candidate
    #    given as the difference, relative
    #    """
    #    mention_tokens = set(normalized(self.mc['text']))
    # candidate_name = self.kb.companies.name.get(self.mc['candidate'])  #get name by id
    #    candidate_tokens = set(normalized(candidate_name))
    #    candidate_tokens_cnt = len(candidate_tokens)
    #    common = mention_tokens.intersection(candidate_tokens)
    #    ratio = len(common) / candidate_tokens_cnt
    #    return abs(ratio)
    def mc_country(self):
        domicile = self.kb.companies.domicile.get(self.mc['candidate'])
        if domicile is None:
            return None
        country_code = COUNTRY_CODES['from_name'].get(domicile.lower())
        country_cnts = self.mc['doc_countries']
        return country_cnts.get(country_code, 0)

    def mc_jaro_distance(self):
        return jf.jaro_distance(self.mention_text.lower(), self.candidate_name.lower())

    # def mc_jaro_winkler(self):
    #    return jf.jaro_winkler(self.mention_text, self.candidate_name)
    # def mc_levenshtein_distance(self):
    #    return jf.levenshtein_distance(self.mention_text, self.candidate_name)
    # def mc_is_surfaceform(self):
    #    """True if the given candidate is a surface form of mention"""
    #    if self.mention_text in self.kb.id2surface_forms.get(self.mc['candidate'], {}):
    #        return True
    #    return False
    def mc_is_exact_match(self):
        """True if the given candidate is an exact match"""
        candidate_name = self.kb.companies.name.get(self.mc['candidate'], '')
        if self.mention_text.lower() == candidate_name.lower():
            return True
        return False

    def mc_is_close_match(self):
        """True if the given candidate is a close match
        only missing a word like Inc or Ltd
        """
        stripped = list(
            self.kb.common_stripped.get(self.mc['candidate'], ['']))[0]
        if jf.jaro_winkler(self.mention_text, stripped) > 0.95:
            return True
        return False

    # Too slow
    # def mc_tfidf(self):
    #    mention_vect = self.kb.tfidf_vect.transform([self.mention_text])
    #    candidate_vect = self.kb.tfidf_vect.transform([self.candidate_name])
    #    result = linear_kernel(mention_vect, candidate_vect).flatten()[0]
    #    return result if not pd.isnull(result) else 0

    #**********************************************************#
    #    Required for use in scikits Pipeline, FeatureUnion
    #**********************************************************#
    def transform(self, mentions, **transform_params):

        if isinstance(mentions, pd.DataFrame):
            mentions = mentions.to_dict(outtype='records')

        all_features = []
        for mention in mentions:
            _feats = MentionCandidateFeatures(self.kb, self.stories, mention)
            if self.append_features:
                try:
                    mention.update(_feats.features)
                    all_features.append(mention)
                except AttributeError:
                    print mention
            else:
                all_features.append(_feats.features)
        df = pd.DataFrame(all_features)
        #**********************************************************#
        #              Document aggregate features
        #**********************************************************#
        # Calculate mention counts per document
        df['mc_count'] = df.groupby(
            ['story_id', 'candidate'])['candidate'].transform('count')

        # normalize counts per document
        def normalize_counts(grp):
            grp['mc_count'] = grp['mc_count'] / grp['mc_count'].max()
            return grp
        df = df.groupby('story_id').apply(normalize_counts)

        # normalize tfidf within candidates
        # def normalize_tf_idf(grp):
        #    s = grp['mc_tfidf'].sum()
        #    if s != 0:
        #        grp['mc_tfidf'] = grp['mc_tfidf'] / s
        #    return grp
        #df = df.groupby(['story_id', 'text']).apply(normalize_tf_idf)
        return df  # .to_dict(outtype='records')

    def fit(self, X, y=None, **fit_params):
        return self

class ModelWrapper(TransformerMixin):
    """Wraps a sklearn algo for use in Pipeline
    """
    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X))


class Numerizer(TransformerMixin):

    """Pipeline transform extract all numeric features
    from a dataframe and prepare them for the ML models. If
    initialized with `stories` then the target `y` is extracted
    from the stories's tags and returned"""

    def __init__(self, stories=None, df=None):
        self.stories = stories
        if self.stories is not None:  # training set
            candidates = df[['candidate', 'story_id']].to_dict('records')
            y = np.array([get_relevance(candidate, self.stories)
                          for candidate in candidates])
            self.target = pd.get_dummies(y)
            self.y = y
            self.sample_weight = balance_weights(y)

    def transform(self, df, **transform_params):
        df = df.copy(deep=True)
        for column in df.columns:
            if df[column].dtype.name == 'object':
                df.drop(column, axis=1, inplace=True)
                continue
            if df[column].dtype.name == 'bool':
                df[column] = df[column].astype(int)

        # Extract numeric but irrelevant columns
        for col in ['candidate', 'story_id']:
            if col in df.columns:
                df.drop(col, 1, inplace=True)
        self.columns = df.columns
        X = df.values
        X = StandardScaler().fit_transform(X)
        if self.stories is not None:
            return X
        return X

    def get_params(self, deep=False):
        return {'stories': self.stories}

    def fit(self, X, y=None, **fit_params):
        return self


class TagFreqs(TransformerMixin):

    """Adds a column with the company tag frequencies as found
    in the training set. Must be initialized with the training set.
    """

    def __init__(self, stories):
        self.freqs = Counter(tag for story in stories[:100]
                             for tag in story['tags'])

    def transform(self, df, **transform_params):
        df['tagged_cnt'] = df.candidate.apply(lambda x: self.freqs.get(x, 0))
        return df

    def fit(self, X, y=None, **fit_params):
        return self


class FeatureSelector(TransformerMixin):

    """Selects given features
    Example:
        FeatureSelector(['text', 'paragraph']).transform(mentions)
    """

    def __init__(self, features):
        self.features = features

    def transform(self, mentions, **transform_params):
        df = pd.DataFrame(mentions)[self.features]
        return df  # .to_dict(outtype='records')

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return {'stories': self.stories}


class PipelinePrint(TransformerMixin):

    """Prints a message ina Pipeline
    """

    def __init__(self, msg):
        self.msg = msg

    def transform(self, mentions, **transform_params):
        print msg
        return mentions  # .to_dict(outtype='records')

    def fit(self, X, y=None, **fit_params):
        return self

# Usage example
if __name__ == "__main__":
    from utils import Companies, Stories
    from knowledgebase import KnowledgeBase
    from lookuptagger import LookupTagger

    companies = Companies.from_xml('data/CompanyList.xml')
    kb = KnowledgeBase(
        companies, 'data/surface_forms.pickle', 'data/tickers.pickle')
    stories = Stories.from_xml('data/train.xml')

    lookup_tagger = LookupTagger(kb)
    mentions = lookup_tagger.fit_transform(stories[:2])
    # Extract mention features for all mentions in the first story
    df = MentionFeatures(kb, stories._token_freqs).transform(mentions[:2])
    print df.head()
