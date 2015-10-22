from __future__ import division
import xml.etree.cElementTree as ET
from xml.etree.cElementTree import Element, SubElement
from datetime import datetime
import pandas as pd
import re
import nltk
from collections import Counter
import cytoolz
from cytoolz import unique, count
from collections import defaultdict, namedtuple
import jellyfish
import cPickle as pickle
from ftfy import fix_text
import joblib
import os

#CountryCode = namedtuple('CountryCode', 'from_name from_city')
#globals()[CountryCode.__name__] = CountryCode
#locals()[CountryCode.__name__] = CountryCode

package_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(package_dir, '..', 'data', 'country_code.pickle')

#**********************************************************#
#                   Import data
#**********************************************************#


COUNTRY_CODES = joblib.load(filename)


filename = os.path.join(package_dir, '..', 'data', 'paragraph_blacklist.txt')
with open(filename) as f:
    DISCLAIMERS = set(line.rstrip('\n') for line in f
                      if not line.startswith('#'))


class XMLParser(list):

    """Wrapper class for he XML parser, esentially a list of elements.
    Also some utilities for list of strings like tokenization.
    Child classes should define at least:
        `_root_tag`, string, All children of this element will be parsed
        `_process_tag`, static method with signature ElementTree->dict
    """

    _root_tag = None
    _process_tag = None

    def __init__(self, data):
        super(XMLParser, self).__init__(data)

    @classmethod
    def from_xml(cls, filename):
        """Parse the XML contents from filename"""

        items = ET.parse(filename).findall(cls._root_tag)
        data = [cls._process_tag(item) for item in items]
        return cls(data)

    @classmethod
    def from_df(cls, df):
        return cls(df.to_dict())

    def to_df(self):
        return pd.DataFrame(self)

    def to_xml(self):
        raise NotImplementedError

    #**********************************************************#
    #        Common methods for Stories and Companies
    #**********************************************************#
    def tokenize_items(self, text_field, lowercase=False):
        return [normalized(item[text_field], lowercase=lowercase)
                for item in self]

    def token_list(self, text_field, lowercase=False):
        """Given a an iterable of strings, tokenize
        and join all tokens in a list
        """

        return [token for item in self
                for token in nltk.word_tokenize(item[text_field])]

    def token_freqs(self, text_field, normalized=True):
        """Calculates the (normalized) frequencies
        """
        token_list = [token.lower()
                      for token in self.token_list(text_field)]
        freqs = Counter(token_list)
        if normalized:
            _max = freqs.most_common(1)[0][1]
            for token in freqs:
                freqs[token] = freqs[token] / _max
        return freqs


class Stories(XMLParser):

    _root_tag = './/Story_Identifier'

    def __init__(self, data):
        super(Stories, self).__init__(data)
        self.id2story = {story['story_id']: story for story in data}
        self.token_freqs = self.token_freqs('text')

    @staticmethod
    def _process_tag(story):
        tags = story.find('Editor_Tags')
        return {'date': datetime.strptime(story.get('Date'), '%d %b %Y'),
                'story_id': int(story.get('StoryId')),
                'headline': fix_text(unicode(story.find('Headline').text)),
                'text': fix_text(unicode(story.find('Story_text').text)),
                'source': story.find('Source').text,
                'tags': {int(e.attrib.get('OrgID')): e.attrib.get('Relevance') for e in tags}
                         if tags is not None else []
                }

    def to_xml(self, filename, id2name):
        results_tag = Element('Results')
        dataset_tag = SubElement(results_tag, 'Dataset')
        dataset_tag.text = 'Test'
        stories_tag = SubElement(results_tag, 'Stories')

        for story in self:
            story_elem = SubElement(
                stories_tag, 'Story', {'StoryId': str(story['story_id'])})
            tags = SubElement(story_elem, 'Tags')
            for tag in story['tags']:
                SubElement(tags, 'Tag',
                           {'OrgID': str(tag),
                            'Company': id2name[tag],
                            'Relevance': "H"})
        ET.ElementTree(results_tag).write(filename)

    def by_id(self, id):
        """Returns a story given its story_id"""
        return self.id2story.get(id)


class Companies(XMLParser):

    _root_tag = './/Company'

    def __init__(self, data):
        super(Companies, self).__init__(data)
        self.name = {company['id']: company['name'] for company in data}
        self.domicile = {company['id']: company['domicile']
                         for company in data}
        self.token_list = self.token_list('name', lowercase=False)

    @staticmethod
    def _process_tag(company):
        return {'id': int(company.find('OrgID').text),
                'name': fix_text(unicode(company.find('Full_Name').text)),
                'domicile': fix_text(unicode(company.find('Domicile').text))}

    def by_id(self, id):
        """Returns a company given its id"""
        return self.name.get(id)


class Document(unicode):

    """Represents a document and provides a way to iterate Paragraphs
        Example:
            document = Document("...")
            for paragraph in document:
                # ...

    """
    def __new__(cls, headline, body):
        headline = fix_text(unicode(headline))
        body = fix_text(unicode(body))
        obj = unicode.__new__(cls, body)
        obj.headline = headline
        return obj

    def __iter__(self):
        yield Paragraph(self.headline)  # consider headline as paragraph 0

        paragraph_sep = '\n\n\t\t\t\t'
        for text in re.split(paragraph_sep, self):
            yield Paragraph(text)


class Paragraph(unicode):

    """Represents a Paragraph, provides related utilities
    and a way to iterate Sentences.
    """

    def __iter__(self):
        for sentence in nltk.sent_tokenize(self):
            yield sentence

    def is_disclaimer(self):
        """Finds out if the paragraph is a disclaimer-like text
        """
        for first_words in DISCLAIMERS:
            if self.startswith(first_words):
                return True
        return False


class Mention(dict):

    """Represents a possible appearance of a company in a text
    """

    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__ = self

    def _repr_html_(self):
        tbl_fmt = '''
            <table style= "font-size:13px;">
                <tr>
                  {}
                </tr>
                <tr>
                  {}
                </tr>
            </table>'''

        headings = ''.join('<th>{}</th>'.format(k) for k in self)
        data = ''.join('<td>{}</td>'.format(v) for k, v in self.items())
        return tbl_fmt.format(headings, data)

    def __str__(self):
        fields = ['previous_word2', 'previous_word',
                  'text', 'next_word', 'next_word2']
        return '|'.join(self.get(field) for field in fields)


def invert_mapping(d):
    """Invert the mapping from dict `d`
    (Ex: id->set(tickers))
    Returns:
        a dict with ticker->set(ids), or surface_form->set(ids)
    """

    results = defaultdict(set)
    for id, v in d.iteritems():
        for each in v:
            results[each].add(id)
    return results


def pickle_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def window_iter(iterable, window_size):
    """Iterates 'iterable' in n-tuples
    Example:
        >>> window_iter(["Change", "in", "fiscal", "year"], 5)
        [(None, None, 'Change', 'in', 'fiscal'),
         (None, 'Change', 'in', 'fiscal', 'year'),
         ('Change', 'in', 'fiscal', 'year', None),
         ('in', 'fiscal', 'year', None, None)]"""

    padding = int((window_size - 1) / 2)
    return nltk.ngrams(iterable, window_size,
                       pad_left=True, pad_right=True,
                       pad_symbol='NONE')[padding:-padding]


def token_frequencies(stories):
    """Calculate token frequencies for all stories
    """

    c = Counter()

    for story in stories:
        for paragraph in Document(story['headline'], story['text']):
            for sentence in paragraph:
                c['SENTENCE_START'] += 1
                tokens = nltk.word_tokenize(sentence)
                c += Counter(tokens)
                c['SENTENCE_END'] += 1
    return c


def unique_mentions_per_word(mentions, field):
    """Count of unique mentions per previous/next-word
    Parameters:
        mentions, list: a list of Mention objects
        field, string : can be one of `('previous_word', 'next_word')`
    Returns:
        a dictionary with words as keys and counts as values
    """
    d = defaultdict(int)
    groups = cytoolz.groupby(lambda x: x[field], mentions)
    for k, g in groups.iteritems():
        d[k] = count(unique(g, lambda x: x.text))

    return d


def token_list(l):
    """Given a an iterable of strings, tokenize
    and join all tokens in a list"""
    return [token for each in l
            for token in nltk.word_tokenize(fix_text(unicode(each)))]


def token_freqs(token_list, normalized=True):
    """Calculates the (normalized) frequencies for all tokens in token_list.
    """
    freqs = Counter(token_list)
    if normalized:
        _max = freqs.most_common(1)[0][1]
        for token in freqs:
            freqs[token] = freqs[token] / _max
    return freqs


import inspect


class memoized(object):

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args, **kwargs):
        key = self.key(args, kwargs)
        if key not in self.cache:
            self.cache[key] = self.func(*args, **kwargs)
        return self.cache[key]

    def normalize_args(self, args, kwargs):
        spec = inspect.getargs(self.func.__code__).args
        return dict(kwargs.items() + zip(spec, args))

    def key(self, args, kwargs):
        a = self.normalize_args(args, kwargs)
        return tuple(sorted(a.items()))


@memoized
def normalized(text, lowercase=True, fix=True, tuples=False):
    """Tokenize, remove capitalization and exclude punctuation
    """
    if fix:
        text = fix_text(unicode(text))
    pattern = r'''(?x)    # verbose regexps
        \w+(-\w+)*        # words with optional internal hyphens
    '''
    result = [w for w in nltk.regexp_tokenize(text, pattern)]
    if lowercase:
        result = [w.lower() for w in nltk.regexp_tokenize(text, pattern)]
    if tuples:
        result = tuple(result)
    return result


def f1_score(pred_tags, true_tags):
    """Calculate the f1_score given two lists of tags"""
    pred_tags, true_tags = set(pred_tags), set(true_tags)
    tp = len(true_tags & pred_tags)
    if tp <= 0:
        return 0
    fp = max(len(pred_tags) - tp, 0)
    fn = len(true_tags) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def multi_replace(regexes, text):
    """Replaces multiple regular expressions in one pass

    regexes: a collection of strings in the format:
        '(?P<REPLACEMENT>REGEX)'"""
    regex = re.compile("|".join(regexes), re.DOTALL)

    def _sub(mo):
        for k, v in mo.groupdict().iteritems():
            if v and k != 'EMPTY':
                return ' ' + k + ' '

    return regex.sub(_sub, text)


def strip_common(text):
    """Strips common forms from a company

    Example:
        >>> strip_common('Apple Inc')
        Apple
        >>> strip_common('CommonWealth REIT Co')
        'CommonWealth REIT'
        >>> strip_common('Cognosante LLC')
        'Cognosante'
    """
    common = set([' Ltd', ' Inc', ' Co', ' Corp', ' SA', ' eG', ' Group',
                  ' Holdings', ' PLC', ' LLC', ' AG', ' LP', ' Investment',
                  ' Company', ' AB', ' Bhd'])

    for each in common:
        if each in text:
            return text.replace(each, '').strip()
    return text


def initials(text):
    """Returns the initials of a name

    Example:
        >>> initials('Federal Home Loan Mortgage Corp')
        FHLMC
    """
    return ''.join(w[0].upper() for w in normalized(text))


def multi_anotate(regexes, text):
    """Add tags to regular expressions in one pass

    regexes: a collection of strings in the format:
        '(?P<REPLACEMENT>REGEX)'"""
    regex = re.compile("|".join(regexes), re.DOTALL)

    def _sub(mo):
        for k, v in mo.groupdict().iteritems():
            if v and k != 'EMPTY':
                return u' {} {} '.format(v, k)

    return regex.sub(_sub, text)


def df_order(df, front_cols):
    back_cols = [c for c in df.columns if c not in front_cols]
    return df[front_cols + back_cols]

from xml.etree import ElementTree
from xml.dom import minidom


def text_countries(text, country_code=COUNTRY_CODES):
    """Given a news text find the possible countries mentioned.
    Parameters:
        text, string: The news text
        country_code, dict:
            A dict with two subdicts mapping name->code and city->code
    Returns:
        A Counter object with the country codes as keys and normalized
        counts as values.
    Example:
        >>> text_countries("MCLEAN, Va.", country_code)
        Counter({'US': 1.0})
    """

    cnt = Counter()
    city_regex = r"[A-Z]+[a-z]*(?:[ '-][A-Z]+[a-z]*)*"
    candidates = re.findall(city_regex, text)
    for candidate in candidates:
        candidate = candidate.lower()
        if candidate in country_code['from_name']:
            code = country_code['from_name'][candidate]
            cnt[code] += 1
            continue
        if candidate in country_code['from_city']:
            code = country_code['from_city'][candidate]
            cnt[code] += 1
    # normalize
    total = sum(cnt.values(), 0.0)
    for key in cnt:
        cnt[key] /= total
    return cnt


def update_dict_set(d, new_elements):
    """Adds elements to each key of dict `d` with
    elements of `new_elements`

    Parameters:
        d: defaultdict(set): this will be updated
        new_elements, defaultdict(set): elements
            to be added.

    Example:
        >>> d['a'] = set([1,2,3])
        >>> new_elements['a'] = set([4])
        >>> update_dict_set(d, new_elements)
        defaultdict(<type 'set'>, {'a': set([1, 2, 3, 4])})

    Returns:
        the original dict with updated keys
    Note: dict is also modified in place
    """

    for k in set(d.keys()).union(set(new_elements.keys())):
        if new_elements.get(k, None):
            d[k] = d[k].union(new_elements[k])
    return d


def normalize_counter(rec, weight=1):
    """Normalizes each recommendation proportionally to the score of the leading tag"""
    try:
        max_score = rec[max(rec, key=rec.get)]
        for tag in rec:
            rec[tag] = weight * rec[tag] / max_score
    except:
        rec = Counter()    # empty recommendation
    return rec


def to_xml(filename, stories, companies):
    """Export fo XML file in the required format

    Parameters:
        filename, string: Ex: 'submision.xml'
        stories, list: A list of dicts
        companies: should provide a `id->companies.name` mapping

    Example:
        >>> to_xml('output.xml', test_stories, companies)
    """

    results_tag = Element('Results')
    dataset_tag = SubElement(results_tag, 'Dataset')
    dataset_tag.text = 'Training'
    stories_tag = SubElement(results_tag, 'Stories')

    for story in stories:
        story_elem = SubElement(
            stories_tag, 'Story', {'StoryId': str(story['story_id'])})
        tags = SubElement(story_elem, 'Tags')
        for org_id in story['tags']:
            SubElement(tags, 'Tag',
                       {'OrgID': str(org_id),
                        'Company': companies.name[org_id],
                        'Relevance': "H"})
    ET.ElementTree(results_tag).write(filename)


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
