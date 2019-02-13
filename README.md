A toy natural language processing example using nltk, textblob, gensim, and spacy.

This project shows how to create a simple API with Flask hosted on the platform. The base URL of the deployed app redirects to a Swagger UI describing the API for potential consumers of the API.

#### Implemented endpoint and arguments:

| endpoint         | query parameters | description                              | possible values                          |
| :--------------- | ---------------- | ---------------------------------------- | ---------------------------------------- |
| /tokenize        | unique           | only return unique tokens?               | true, false (default) (not case sensitive) |
|                  | library          | which library to use for tokenization?   | textblob (default), gensim, spacy, nltk  |
| /sentiment       | kind             | which implementation of sentiment analysis to use (1 = `PatternAnalyzer` from the `pattern` library, 2 = Naive Bayes classifier from `nltk`) | 1 (default), 2 (note: takes a _long_ time to run) |
| /parts_of_speech |                  | return verbs, numbers, nouns, adjectives, and adverbs |                                          |
| /top_k           | k                | return the `k` most common words         | int                                      |
|                  | library          | which library to use?                    | textblob (default), gensim, spacy, nltk  |
| /topics          | num_topics       | Number of topics to return               | int (default = 5)                                      |

#### Examples of usage

- app is deployed
- token has been generated and stored as the environmental variable `TOKEN`
- example text files are based on [this](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences) dataset

tokenize plain text
`curl -X GET "https://enterprise.demo.continuum.io:30090/tokenize?library=nltk" -H "Authorization: Bearer "$TOKEN"" -H "Content-type: text/plain" -d "the quick brown fox jumped over the lazy dog"`

tokenize JSON array
`curl -X GET "https://enterprise.demo.continuum.io:30090/tokenize" -H "Authorization: Bearer "$TOKEN"" -H "Content-type: application/json" -d '{"string_1": "The movie was great. I loved it!", "string_2" : "I did not like the movie at all."}'`

get parts of speech
`curl -X GET "https://enterprise.demo.continuum.io:30090/parts_of_speech" -H "Authorization: Bearer "$TOKEN"" -H "Content-type: text/plain" -d "the quick brown fox jumped over the lazy dog"`

find top 10 words used in negative amazon reviews of cellphone accessaries
`curl -H "Content-type: text/plain" -H "Authorization: Bearer "$TOKEN"" -X GET "https://enterprise.demo.continuum.io:30090/top_k?k=10" --data-binary "@negative_amazon.txt"`

get sentiment analysis of all the negative amazon reviews
`curl -H "Content-type: text/plain" -H "Authorization: Bearer "$TOKEN"" -X GET "https://enterprise.demo.continuum.io:30090/sentiment" --data-binary "@negative_amazon.txt"
`

perform Latent Dirichlet Allocation on the positive reviews and print words the describe the top 4 topics
`curl -H "Content-type: text/plain" -H "Authorization: Bearer "$TOKEN"" -X GET "https://enterprise.demo.continuum.io:30090/topics?num_topics=4" --data-binary "@pos.txt"
`
