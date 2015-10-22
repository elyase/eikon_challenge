from __future__ import division
from utils import *
from lookuptagger import extract_mentions
from sklearn.base import TransformerMixin


class CandidateGenerator(TransformerMixin):

    def __init__(self, kb):
        self.kb = kb

    def transform(self, mentions, **transform_params):
        if isinstance(mentions, pd.DataFrame):
            mentions = mentions.to_dict(outtype='records')

        result = []
        # print mentions[:1]
        for mention in mentions:
            recs = generate_candidates(mention, self.kb)
            mention.update({'candidates': recs})
            for id, weight in recs.iteritems():
                z = mention.copy()
                z.update({'candidate': id,
                          'candidate_weight': weight,
                          'candidate_name': self.kb.companies.name[id]})
                result.append(z)
        # for mention in mentions:
        # print '------', mention['text']
        #    candidate_ids = generate_candidates_old(mention, self.kb)
        #    for id, token in candidate_ids:
        #        mention.update({'candidate': id,
        #                        'origin_token': token,
        #                        'candidate_name': self.kb.companies.name[id]})
        # result.append(mention)
        # print l
        return pd.DataFrame(result)

    def fit(self, X, y=None, **fit_params):
        return self


_mention_cache = defaultdict(Counter)


def generate_candidates(mention, kb, cut_off=(3, 0.2)):
    """Generate candidates for a mention by trasversing its tokens
    and finding corresponding authorities in the KnowledgeBase

    Parameters:
        mention, dict: most have a 'text' field"
        kb, KnowledgeBase: KB object
    """
    # print mention['story_id'], mention['text']

    if mention['text'] in _mention_cache:
        return _mention_cache[mention['text']]
    exact_match = kb.full_names_lower.get(mention['text'].lower(), None)
    if exact_match:
        return Counter({exact_match: 10})

    stripped_match = kb.common_stripped2id.get(mention['text'].lower(), None)
    if stripped_match:
        return Counter({stripped_match: 2})

    surface_form = kb.surface_forms.get(mention['text'], None)
    if surface_form:
        cnt = Counter()
        for coid in surface_form:
            cnt[coid] = 1
        return cnt

    initials = kb.initials.get(mention['text'], None)
    if initials:
        cnt = Counter()
        for coid in initials:
            cnt[coid] = 1
        return cnt

    ticker = kb.tickers.get(mention['text'], None)
    if ticker:
        cnt = Counter()
        for coid in ticker:
            cnt[coid] = 1
        return cnt

    # Each token in a mention recommends several candidates
    # All candidates combined according to the number of appearances
    # weighted by token relevance
    cnt = Counter()
    for token in normalized(mention['text'], lowercase=True, fix=False):
        ids = kb.lowercase_token2coid.get(token, [])
        if len(ids) != 0:
            weight = 1 / len(ids)
        else:
            weight = 0
        for id in ids:
            cnt[id] += weight

    # Filter mentions by total token relevance
    filtered = Counter()
    most_common = cnt.most_common(cut_off[0])
    if not most_common:
        return Counter()

    max_val = most_common[0][1]
    for i, (coid, value) in enumerate(most_common):
        if value > 0.80 * max_val:
            filtered[coid] = value

    _mention_cache[mention['text']] = filtered
    return filtered


def generate_candidates_old(mention, kb):
    """[DEPRECATED]
    Generate candidates for a mention by trasversing its tokens
    and finding corresponding authorities in the KnowledgeBase

    Parameters:
        mention, dict: most have a 'text' field"
        kb, KnowledgeBase: KB object
    """

    candidates = []
    for token in normalized(mention['text']):
        ids = kb.token2coid.get(token, [])
        for id in ids:
            candidates.append((id, token))
    return set(candidates)


def get_relevance(candidate, stories):
    """Get the relevance for a candidate
    """

    # first get the tags for the story
    # where the mention was found
    tags = stories.by_id(candidate['story_id'])['tags']
    # find the relevance for the corresponding candidate
    return tags.get(candidate['candidate'], 'NO')


def is_tagged_mention(mention, stories, kb):
    """[DEPRECATED]
    Return True if mention is tagged in a specific story
    This function returns (hopefully) correct returns in
    the majority of cases.
    """
    story = stories.by_id(mention['story_id'])
    if not story:       # story is test set
        return False
    train_tags = set(story['tags'])
    candidates = generate_candidates(mention, kb)
    intersection = train_tags.intersection(candidates)
    return True if intersection else False


def is_tagged_candidate(candidate, stories):
    """Given a candidate, find the tags in the story it came from
    and check wether it was tagged or not"""
    return candidate['candidate'] in stories.by_id(candidate['story_id'])['tags']


def generate_train_data(stories, kb, target_col='y'):
    """[DEPRECATED]
    Extract mentions from all stories. For each mention a
    boolean `target_col` is calculated representing wether the
    mention belongs to its corresponding story
    """

    mentions = []
    for story in stories:
        for mention in extract_mentions(story, kb):
            mention[target_col] = is_tagged_mention(mention, stories, kb)
            mentions.append(mention)
    return pd.DataFrame(mentions)
