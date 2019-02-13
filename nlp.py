""" A simple REST API that implements several natural language
processing tools like tokenization and sentiment analysis.  """
from flask import Blueprint, Flask, jsonify, url_for, redirect, request
from argparse import ArgumentParser
import sys
import string
import re
from operator import itemgetter
from werkzeug.contrib.fixers import ProxyFix

# for natural language processing
import gensim
from collections import defaultdict, Counter
from gensim.parsing.preprocessing import STOPWORDS
from spacy.lang.en import English
from spacy.attrs import ORTH

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def get_tokens(raw, length=False, unique=False, library='gensim'):
    """Returns a dictionary of tokens from a dictionary of strings."""
    token_dict = {}

    # parse the library
    for key, value in raw.items():
        if library == 'gensim':
            tokens = list(gensim.utils.tokenize(value.lower()))
        elif library == 'spacy':
            parser = English()
            parsed_raw = parser(value.lower())
            # also need to remove punctuation here
            tokens = [str(token)
                      for token in parsed_raw if token.pos_ not in ['PUNCT']]
        elif library == "nltk":
            tokens = nltk.word_tokenize(value.lower())
            # remove punctuation
            tokens = [
                token for token in tokens if token not in string.punctuation]

        # parse length and uniqueness
        if unique and length:
            token_dict[key] = len(set(tokens))
        elif not unique and length:
            token_dict[key] = len(tokens)
        elif not unique and not length:
            token_dict[key] = list(tokens)
        else:
            token_dict[key] = list(set(tokens))

    return token_dict

# ideally add more implementations here


def get_sentiment(string_dict):
    """Return the sentiment score from a dictionary of strings."""
    processed = {}
    sid = SentimentIntensityAnalyzer()
    for key, value in string_dict.items():
        processed[key] = sid.polarity_scores(value)
    return processed


def get_part_of_speech(raw):
    """Return the parts of speech for each string in a dictionary."""
    processed = {}

    for key, value in raw.items():
        tags = nltk.pos_tag(nltk.word_tokenize(value))

        nouns = [word for word, pos in tags if pos in [
            'NN', 'NNS', 'NNP', 'NNPS']]
        adverbs = [word for word, pos in tags if pos in [
            'RB', 'RBR', 'RBS', 'RP']]
        verbs = [word for word, pos in tags if pos in [
            'VB', 'VBZ', 'VBP', 'VBD', 'VBN', 'VBG']]
        numbers = [word for word, pos in tags if pos in ['CD']]
        adjectives = [word for word,
                      pos in tags if pos in ['JJ', 'JJR', 'JJS']]

        processed[key] = {
            'nouns': nouns,
            'adverbs': adverbs,
            'verbs': verbs,
            'numbers': numbers,
            'adjectives': adjectives
        }
    return processed


def parse_data(r):
    """Parse data from request. Return a dictionary of strings."""
    processed = dict()
    if r.headers['Content-Type'] == 'text/plain':
        for number, line in enumerate(r.data.splitlines()):
            name = "string_{:05d}".format(number)
            processed[name] = line.decode("utf-8")
        return processed
    elif r.headers['Content-Type'] == 'text/csv':
        # parse csv file
        # need something more complex here
        # can't just split on , because string should be able to have commas in them
        # maybe use csv module?
        # reader = csv.reader(r.data.decode('utf-8'))
        # for row in reader:
        #     print(row)
        for number, line in enumerate(r.data.splitlines()):
            if number != 0:
                # see here: https://stackoverflow.com/questions/79968/split-a-string-by-spaces-preserving-quoted-substrings-in-python
                pieces = [p for p in re.split(
                    "( |\\\".*?\\\"|'.*?')", line.decode('utf-8')) if p.strip()]
                # assume the last column contains the string
                # assume the 2nd to last column contains the string name (in case someone, e.g. exports from R with row names)
                # line = re.findall(r'"([^"]*)"', line.decode("utf-8"))
                key = "".join(pieces[-2:-1])
                processed[key.strip()] = str(pieces[-1]).strip()
        return processed

    elif r.headers['Content-Type'] == 'application/json':
        for key, value in r.json.items():
            processed[key] = value
        return processed
    else:
        return ""


def get_all_text(raw, lib='nltk', result="list"):
    """Return a list with **all** the tokens in a group of strings. Excludes stop words."""
    tokens = get_tokens(raw, library=lib)
    x = []
    s = ""
    for key, value in tokens.items():
        for token in value:
            if token.lower() not in STOPWORDS:
                if result == "list":
                    x.append(token)
                else:
                    s = " ".join([s, token])
    if result == "list":
        return(x)
    else:
        return(s)


def get_top_k(r, k, library='nltk'):
    """Get the k most common tokens in a dictionary of strings."""

    # if using nltk:
    if library == 'nltk':
        # do the tokenization, return all the text from the strings stored in the dict
        x = get_all_text(r, lib='nltk')
        # remove stopwords
        # filtered_words = [word for word in x if not word in stopwords.words('english')]
        counts = Counter(x)
        return(counts.most_common(int(k)))

    elif library == 'spacy':
        x = get_all_text(r, lib='spacy', result="string")
        # x = ""
        # for key, value in r.items():
        #     x = "".join([x, value])
        lang = English()
        doc = lang(x)
        # get the counts for each word
        counts = doc.count_by(ORTH)
        # now get the top k ids and find their word equivalents
        dist = []
        for word_id, count in sorted(counts.items(), reverse=True, key=lambda item: item[1]):
            # create a list of tuples because that's what nltk.FreqDist.most_common produces
            dist.append((lang.vocab.strings[word_id], count))
        # return the top n counts and words
        return(dist[:k])
    elif library == "gensim":
        x = get_all_text(r, lib="gensim", result="list")
        frequency = defaultdict(int)
        for token in x:
            frequency[token] += 1
        out = []
        for word, count in sorted(frequency.items(), reverse=True, key=itemgetter(1)):
            out.append((word, count))
        return(out[:k])
    else:
        return('library not found')


def get_topics(r, number_of_topics):
    '''Implement Latent Dirichlet Allocation. Return 5 topics from all text passed to the API.'''

    # tokenize
    tokens = get_tokens(r)

    # `tokens` is a dict; we want a list and name it `docs`:
    docs = []
    for key, value in tokens.items():
        docs.append(value)

    # remove stopwords using stopwords from nltk
    docs_cleaned = [[token for token in doc if token not in STOPWORDS] for doc in docs]

    # create dictionary
    dictionary = gensim.corpora.Dictionary(docs_cleaned)

    # create document matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs_cleaned]

    ldamodel = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=number_of_topics,
                                               id2word=dictionary, passes=50)

    return(ldamodel.print_topics(number_of_topics))


# set up the app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.config['PREFERRED_URL_SCHEME'] = 'https'
app.config['project_hosts'] = []

bp = Blueprint('main', __name__)

# define endpoints


@bp.route('/')
def get_swagger():
    """Show Swagger UI."""
    return redirect(url_for('static', filename='index.html'))


@bp.route('/tokenize', methods=['POST'])
# curl -H "Content-type: application/json" -X POST http://0.0.0.0:8086/tokenize -d '{"data": "The movie was great. I loved it!"}'
def return_tokens():
    # parse query parameters
    unique = request.args.get('unique')
    # to accept True, true, and TRUE:
    unique = unique.lower() if unique is not None else unique
    library = request.args.get('library')
    library = 'textblob' if library is None else library

    # parse the data from the request
    raw = parse_data(request)

    if raw == {}:
        return "415 Unsupported Data Input Type"

    # get tokens
    if unique == 'true':
        tokens = get_tokens(raw, unique=True)
    else:
        tokens = get_tokens(raw)

    return(jsonify({'library': library, 'tokens': tokens}))

@bp.route('/sentiment', methods=['POST'])
# ## send JSON: curl -H "Content-type: application/json" -X POST http://0.0.0.0:8086/sentiment -d '{"data": "This movie suckssssss. I hated it so much! SAD"}'
# ## send a test string: curl -H "Content-type: text/plain" -X POST http://0.0.0.0:8086/sentiment -d "I am so happy. Today is a great day\!"
# ## send a test file: curl -H "Content-type: text/plain" -X POST http://0.0.0.0:8086/sentiment --data-binary "@test.txt"
def return_sentiment():

    # parse the data from the request
    raw = parse_data(request)

    if raw == {}:
        return "415 Unsupported Data Input Type"

    return(jsonify({'sentiment_score': get_sentiment(raw)}))


@bp.route('/parts_of_speech', methods=['POST'])
# # for string stored in newlines in text.txt in CWD: curl -H "Content-type: text/plain" -X POST http://0.0.0.0:8086/parts_of_speech --data-binary "@test.txt"
def get_parts_of_speech():
    raw = parse_data(request)

    if raw == {}:
        return "415 Unsupported Data Input Type"

    return(jsonify(get_part_of_speech(raw)))


@bp.route('/top_k', methods=['POST'])
# curl -H "Content-type: text/plain" -X POST http://0.0.0.0:8086/top_k?library=gensim --data-binary "@list_of_strings.txt"
def top_k():
    library = request.args.get('library')
    library = 'gensim' if library is None else library
    k = request.args.get('k')
    k = 5 if k is None else int(k)

    raw = parse_data(request)

    if raw == {}:
        return "415 Unsupported Data Input Type"
    out = get_top_k(raw, k=k, library=library)

    return(jsonify({'library': library, 'k': k, 'top_k': out}))


@bp.route('/topics', methods=['POST'])
def lda():
    num_topics = request.args.get('num_topics')
    num_topics = 5 if num_topics is None else int(num_topics)

    raw = parse_data(request)
    if raw == {}:
        return "415 Unsupported Data Input Type"

    topics = get_topics(raw, num_topics)

    topics_pretty = [re.findall('"([^"]*)"', topic[1]) for topic in topics]

    out = {}
    for num, topic in enumerate(topics_pretty):
        out["topic_" + str(num)] = topic

    return(jsonify({'number_of_topics': num_topics, 'topics': out}))


if __name__ == '__main__':
    # arg parser for the standard anaconda-project options
    parser = ArgumentParser(prog="nlp-api",
                            description="Natural Language Processing API")
    parser.add_argument('--anaconda-project-host', action='append', default=[],
                        help='Hostname to allow in requests')
    parser.add_argument('--anaconda-project-port', action='store', default=8086, type=int,
                        help='Port to listen on')
    parser.add_argument('--anaconda-project-iframe-hosts',
                        action='append',
                        help='Space-separated hosts which can embed us in an iframe per our Content-Security-Policy')
    parser.add_argument('--anaconda-project-no-browser', action='store_true',
                        default=False,
                        help='Disable opening in a browser')
    parser.add_argument('--anaconda-project-use-xheaders',
                        action='store_true',
                        default=False,
                        help='Trust X-headers from reverse proxy')
    parser.add_argument('--anaconda-project-url-prefix', action='store', default='',
                        help='Prefix in front of urls')
    parser.add_argument('--anaconda-project-address',
                        action='store',
                        default='0.0.0.0',
                        help='IP address the application should listen on.')

    args = parser.parse_args(sys.argv[1:])
    project_hosts = args.anaconda_project_host
    app.config['project_hosts'] = project_hosts
    app.config['project_port'] = args.anaconda_project_port
    app.register_blueprint(bp, url_prefix=args.anaconda_project_url_prefix)
    app.run(debug=True,
            port=args.anaconda_project_port, host=args.anaconda_project_address)
