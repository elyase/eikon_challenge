from utils import *
from knowledgebase import *
from lookuptagger import *
from candidates import *


def test_Paragraph():
    text = "THIS PARAGRAPH IS A DISCLAIMER BECAUSE ALMOST EVERYTHING IS UPPERCASE: me"
    for sentence in Paragraph(text):
        continue
    assert Paragraph(text).is_disclaimer()


def test_KnowledgBase():
    companies = [{'domicile': 'United States',
                  'id': 5001350717,
                  'name': '"NORTH AMERICAN AIRCRAFT SERVICES, INC."'},
                 {'domicile': 'Canada',
                  'id': 4295861193,
                  'name': '01 Communique Laboratory Inc'}]
    kb = KnowledgeBase(companies)
    assert '01 Communique Laboratory Inc' in kb.full_names
    assert '"NORTH AMERICAN AIRCRAFT SERVICES, INC."'.lower(
    ) in kb.full_names_lower
    assert kb.is_case_insensitive_exact_match('01 communique Laboratory inc')
    assert not kb.is_case_sensitive_exact_match('01 communique Laboratory inc')
    assert 'american' in kb.lowercase_tokens
    assert kb.company_tokens_freqs['INC'] == 1


def test_window_iter():
    text = "Change in fiscal year"
    result = window_iter(text.split(), 5)
    expected = [('NONE', 'NONE', 'Change', 'in', 'fiscal'),
                ('NONE', 'Change', 'in', 'fiscal', 'year'),
                ('Change', 'in', 'fiscal', 'year', 'NONE'),
                ('in', 'fiscal', 'year', 'NONE', 'NONE')]
    assert result == expected


def test_sentence_mentions():
    KB = namedtuple('KB', 'lowercase_tokens, lower_token_whitelist')
    kb = KB(lowercase_tokens={'goodwin', 'procter', 'llp', 'equity', 'residential',
                              'bank', 'america', 'of'},
            lower_token_whitelist={'of': set([('Bank', 'America')])})
    sentence = 'Goodwin Procter LLP next previous EQUITY RESIDENTIAL and Bank of America'
    mentions = [mention for mention in sentence_mentions(sentence,
                                                         kb, 2, 1, 0)]
    expected = [Mention(text='Goodwin Procter LLP', previous_word2='NONE', previous_word='NONE',
                        next_word='next', next_word2='previous',
                        pos_in_sentence=0.0, paragraph=2, sentence=1, token_cnt=3, story_id=0),
                Mention(text='EQUITY RESIDENTIAL', previous_word2='next', previous_word='previous',
                        next_word='and', next_word2='Bank',
                        pos_in_sentence=0.5555555555555556, paragraph=2, sentence=1, token_cnt=2, story_id=0),
                Mention(text='Bank of America', previous_word2='RESIDENTIAL', previous_word='and',
                        next_word='NONE', next_word2='NONE',
                        pos_in_sentence=1.0, paragraph=2, sentence=1, token_cnt=3, story_id=0)]
    assert mentions == expected


def test_generate_candidates():
    kb = namedtuple('KB', 'token2coid')({'token1': ['id1'],
                                         'token2': ['id1', 'id2'],
                                         'token3': ['id3']})
    mention = Mention(text=u'token1 & (token2)')
    candidates = generate_candidates(mention, kb)
    assert candidates == {'id1', 'id2'}


def test_is_tagged_mention():
    kb = namedtuple('KB', 'token2coid')({'token1': ['id1'],
                                         'token2': ['id1', 'id2'],
                                         'token3': ['id3']})

    class MockStories:

        def by_id(self, x):
            return {'tags': {'id1', 'id2'}}

    stories = MockStories()
    mention = Mention(text=u'token1 & (token2)', story_id='any')
    assert is_tagged_mention(mention, stories, kb)
    mention = Mention(text=u'token4 and token5', story_id='any')
    assert not is_tagged_mention(mention, stories, kb)


def test_token_frequencies():
    stories = [{'headline': 'headline_token1',
                'text': 'text_token1, text_token2 text_token1 text_token1'},
               {'headline': 'headline_token2, headline_token2',
                'text': u'text_token2 text_token2 text_token2'}]

    freqs = token_frequencies(stories)
    assert freqs['headline_token2'] == 2
    assert freqs['text_token2'] == 4
    assert freqs['headline_token1'] == 1
    assert freqs['SENTENCE_START'] == 4
    assert freqs['SENTENCE_END'] == 4


def test_unique_mentions_per_word():
    mentions = [Mention(text='same_mention', previous_word='previous_word1'),
                Mention(text='same_mention', previous_word='previous_word1')]

    assert unique_mentions_per_word(
        mentions, 'previous_word')['previous_word1'] == 1

    mentions = [Mention(text='same_mention', next_word='next_word1'),
                Mention(text='same_mention', next_word='next_word1'),
                Mention(text='other_mention', next_word='next_word1')]
    assert unique_mentions_per_word(mentions, 'next_word')['next_word1'] == 2


def test_normalized():
    texts = ['"NORTH AMERICAN AIRCRAFT SERVICES, INC."',
             '1-800-Flowers.Com Inc',
             '1Malaysia Development Bhd',
             'AIG United Guaranty Insurance (Asia) Ltd',
             'Alba Group plc &amp; Co KG']
    normalized_texts = [normalized(text) for text in texts]
    expected = [['north', 'american', 'aircraft', 'services', 'inc'],
                ['1-800-flowers', 'com', 'inc'],
                ['1malaysia', 'development', 'bhd'],
                ['aig', 'united', 'guaranty', 'insurance', 'asia', 'ltd'],
                ['alba', 'group', 'plc', 'co', 'kg']]
    assert expected == normalized_texts
