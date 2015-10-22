# eikon_challenge


In this challenge Thomson Reuters, was searching for an algorithm to accurately tag incoming news items by relevance for companies or organizations mentioned within the news item. I built a system capable of recognizing alternative company names (using DBpedia data), stock ticker based identification (Bloomber Symbiology data) and country based discrimination in the text of the news. The system has the following structure:

![diagram](http://yasermartinez.com/blog/img/projects_thomson_reuter_diagram.png)

Lookup tagger: Performs authorithy driven mention detection, i.e. extracts with high recall possible mentions of company names.
Candidate generation: For each possible company mention several candidate companies are suggested
Features generation: For each mention-candidate company generate features.
Classifier: This component finds the correct candidates using the features.
One of the greatest challenges was to find data sources to augment the information about the list of companies complying with the accepted licenses.
