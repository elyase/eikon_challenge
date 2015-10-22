from __future__ import division
from collections import namedtuple
CountryCode = namedtuple('CountryCode', 'from_name from_city')
from utils import *
from sklearn.base import TransformerMixin

filename = os.path.join(package_dir, '..', 'data', 'mentions_blacklist.txt')
try:
    with open(filename) as f:
        mention_blacklist = set(line.rstrip('\n') for line in f
                                if not line.startswith('#'))
except:
    mention_blacklist = {}


class LookupTagger(TransformerMixin):

    def __init__(self, kb):
        self.kb = kb

    def transform(self, stories, **transform_params):
        return extract_mentions(stories, self.kb)

    def fit(self, X, y=None, **fit_params):
        return self


def is_company_token(token, prev_word, next_word, kb):
    """True if `token` could be part of a company name
    """
    result = all([token,                                 # not empty/null
                  token.lower() in kb.lowercase_tokens,  # part of KB
                  token[0].isupper() or                  # starts with capital
                  (prev_word, next_word) in kb.lower_token_whitelist.get(token, {})])
    return result


def sentence_mentions(sentence, kb, par_idx=0, sent_idx=0, story_id=0):
    """Extract all mentions from `sentence` using a KnowledgeBase
    Parameters:
        sentence, string:
        kb, KnowledgeBase: mentions are composed of kb.tokens
        par_idx, sent_idx, int: paragraph and sentence index
    """

    mention_tokens = []
    tokens = nltk.word_tokenize(sentence)
    sentence_length = len(tokens)

    for i, (prev2, prev, token, nexxt, nexxt2) in enumerate(window_iter(tokens, 5)):
        # condition to include a token
        if is_company_token(token, prev, nexxt, kb):
            mention_tokens.append(token)
            # first token in mention => save previous tokens
            if len(mention_tokens) == 1:
                previous_word = prev
                previous_word2 = prev2
                pos_in_sentence = i

        is_next_token_company = is_company_token(nexxt, token, nexxt2, kb)
        # token not in KB and not empty
        if (not is_next_token_company and mention_tokens):
            # calculate position in sentence
            mention_length = len(mention_tokens)
            if sentence_length > mention_length:   # normalize
                pos_in_sentence /= sentence_length - mention_length

            text = ' '.join(mention_tokens)
            # Quick patches
            prefixes = set(['ON ', 'And ', 'Of ', 'On ', 'If '])
            for each in prefixes:
                if text.startswith(each):
                    text = text.lstrip(each)

            if not text.lower() in mention_blacklist:
                yield Mention(text=text,
                              previous_word2=previous_word2,
                              previous_word=previous_word,
                              next_word=nexxt,
                              next_word2=nexxt2,
                              pos_in_sentence=pos_in_sentence,
                              paragraph=par_idx,
                              sentence=sent_idx,
                              token_cnt=mention_length,
                              story_id=story_id)
            mention_tokens = []


def extract_mentions(stories, kb):
    """Extract all mentions from `stories` or a single story
    """
    if type(stories) is dict:
        stories = [stories]

    mentions = []

    for story in stories:
        paragraphs = [par for par in Document(story['headline'],
                                              story['text'])]
        paragraph_count = len(paragraphs)
        for par_num, paragraph in enumerate(paragraphs):
            if paragraph.is_disclaimer():
                continue
            if paragraph_count > 1:
                par_num /= paragraph_count - 1       # normalize
            sentences = [sent for sent in paragraph]
            sentence_count = len(sentences)
            for sent_num, sentence in enumerate(sentences):
                if sentence_count > 1:
                    sent_num /= sentence_count - 1   # normalize
                mentions.extend(sentence_mentions(sentence, kb,
                                                  par_num, sent_num,
                                                  story['story_id']))

    return mentions

# Usage example
if __name__ == "__main__":

    from utils import Companies, Stories
    from knowledgebase import KnowledgeBase

    companies = Companies.from_xml('../data/CompanyList.xml')
    stories = Stories.from_xml('../data/train.xml')

    kb = KnowledgeBase(
        companies, '../data/surface_forms.pickle', '../data/tickers.pickle')

    # Extract mention features for all mentions in the first story
    lookuptagger = LookupTagger(kb)
    mentions = lookuptagger.fit_transform(stories[:10])
    print mentions[:2]
