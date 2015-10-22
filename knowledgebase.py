from __future__ import division
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer


class KnowledgeBase(object):

    """This class provides methods to work with Companies"""
    # TODO optimize all comprehensions in one for loop

    def __init__(self, companies,
                 surface_forms_filename=None,
                 tickers_filename=None):
        self.companies = companies
        self.tfidf_vect = TfidfVectorizer()
        self.tfidf_vect.fit([company['name'] for company in companies])
        self.token_list = companies.token_list
        self.company_tokens_freqs = token_freqs(self.token_list)

        self.tokens = set(self.company_tokens_freqs.keys())
        self.full_names = set(company['name'] for company in companies)
        self.full_names_lower = {company['name'].lower(): company['id']
                                 for company in companies}
        self.normalized_companies = set(normalized(company['name'], tuples=True)
                                        for company in companies)
        self.token2coid = self.token2coid()
        self.lower_token_whitelist = self.get_lower_token_whitelist()

        # Company Initials
        self.initials = {initials(company['name']): set([company['id']])
                         for company in companies}
        self.id2initials = {company['id']: set([initials(company['name'])])
                            for company in companies}
        self.tokens.update(self.initials.keys())
        update_dict_set(self.token2coid, self.initials)
        #

        # Alternate names
        self.common_stripped = {company['id']: set([strip_common(company['name'])])
                                for company in companies}
        self.common_stripped2id = {strip_common(company['name']).lower(): company['id']
                                   for company in companies}
        if surface_forms_filename:
            self.id2surface_forms = pickle_load(surface_forms_filename)
            update_dict_set(self.id2surface_forms, self.common_stripped)
            update_dict_set(self.id2surface_forms, self.id2initials)
            self.surface_forms = invert_mapping(self.id2surface_forms)
            self.surface_forms_tokens = invert_id2surface_forms(
                self.id2surface_forms)
            self.tokens.update(self.surface_forms_tokens.keys())
            update_dict_set(self.token2coid, self.surface_forms_tokens)

        # Stock ticker symbols
        if tickers_filename:
            self.id2ticker = pickle_load(tickers_filename)
            self.tickers = invert_mapping(self.id2ticker)
            self.tokens.update(self.tickers.keys())
            update_dict_set(self.token2coid, self.tickers)

        self.lowercase_tokens = set(token.lower() for token in self.tokens)
        self.lowercase_token2coid = {token.lower(): ids
                                     for token, ids in self.token2coid.iteritems()}

    def token2coid(self):
        """Generate a dict company tokens -> company IDs
        """
        d = defaultdict(set)
        for company in self.companies:
            name = company['name']
            tokens = normalized(name, fix=False, lowercase=False)
            for token in tokens:
                d[token].add(company['id'])
        return d

    def get_lower_token_whitelist(self):
        """Generates a dict with the punctuations symbols
        present in company names, with possible next
        words as the corresponding value
        Example:
            '!': set(['Brands', 'Entertainment', 'Inc', '.'])
            '#': set(['12'])
            '$': set(['High', 'TIPS', 'Treasury'])
        """

        whitelist = defaultdict(set)
        for prev, token, nexxt in window_iter(self.token_list, 3):
            if not token[0].isupper():
                whitelist[token].add((prev, nexxt))

        # hacks
        # whitelist[','].add((,,))
        return whitelist


def invert_id2surface_forms(id2surface_forms):
    """Invert the mapping id2surface_forms
    Returns:
        a dict with token->set(ids)
        {u'Aisin': set([4295880739]),
         u'AGL': set([4295903259, 4296432542, 4295858403])
        }
    """

    results = defaultdict(set)
    for id, surface_forms in id2surface_forms.iteritems():
        for surface_form in surface_forms:
            for token in normalized(fix_text(surface_form), lowercase=False):
                results[token].add(id)
    return results


if __name__ == "__main__":
    from utils import Companies, Stories
    from knowledgebase import KnowledgeBase

    companies = Companies.from_xml('../data/CompanyList.xml')
    kb = KnowledgeBase(
        companies, '../data/surface_forms.pickle', '../data/tickers.pickle')
