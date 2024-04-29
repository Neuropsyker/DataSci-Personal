# Loads data files into memory
def load_files(directory):
    result = []
    for fname in os.listdir(directory):
        with open(directory + fname, 'r', encoding='UTF-8', errors='ignore') as f:
            result.append(f.read())
    return result

# Converting words into lists of lower case tokens:
def split_into_words(line):
    word_regex_improved = r"(\w[\w']*\w|\w)"
    word_matcher = re.compile(word_regex_improved)
    return word_matcher.findall(line)

# Applies all preprocessing steps in a single function
def preprocess_sentence(sentence):
    lemmatizer = nltk.WordNetLemmatizer()
    # Processing pipeline:
    processed_tokens = nltk.word_tokenize(sentence)
    processed_tokens = [w.lower() for w in processed_tokens]
    # Find the 10 least common words:
    word_counts = collections.Counter(processed_tokens)
    uncommon_words = word_counts.most_common()[:-10:-1]
    # Execute removal of stopwords and uncommon words:
    processed_tokens = [w for w in processed_tokens if w not in stop_words]
    processed_tokens = [w for w in processed_tokens if w not in uncommon_words]
    # Lemmatize the output so far:
    processed_tokens = [lemmatizer.lemmatize(w) for w in processed_tokens]
    return processed_tokens

# Converts each word token into a feature.
def feature_extractor(tokens):
    return dict(collections.Counter(tokens))

# Applies train/test split at 70:30 ratio.
def train_test_splitter(dataset, train_size = 0.7):
    training_examples = int(len(dataset) * train_size)
    return dataset[:training_examples], dataset[training_examples:]

# Build new feature extractor using a vectorizer, to extract TF-IDF features:
def feature_vectorizer(corpus):
    sa_stop_words = nltk.corpus.stopwords.words('english')
    # Create a list of exceptions, as these stopwords may change a sentence's sentiment if removed.
    sa_white_list = ['what', 'but', 'if', 'because', 'as', 'until', 'against', 'up', 'down', 'in', 'out',
                    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'why',
                    'how', 'all', 'any', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                    'same', 'so', 'than', 'too', 'can', 'will', 'just', 'don', 'should']
    # Remove stop words except for those specified in the white list.
    sa_stop_words = [sw for sw in sa_stop_words if sw not in sa_white_list]
    # Instantiate the vectorizer.
    count_vectorizer = feature_extraction.text.CountVectorizer(
        lowercase = True,
        tokenizer = nltk.word_tokenize,
        min_df=2, # this means the term frequency must be 2 or higher.
        ngram_range=(1,2),
        stop_words=sa_stop_words
    )
    # Run the vectorizer on the body of text ('corpus').
    processed_corpus = count_vectorizer.fit_transform(corpus)
    processed_corpus = feature_extraction.text.TfidfTransformer().fit_transform(processed_corpus)
    return processed_corpus

# Begin pipeline construction and assembly.
# Docstrings added to benefit explainability:
def pipeline(f):
    """
    Decorator function for creating a pipeline.
    Args: #
    f (function): The generator function to be wrapped.
    Returns:
    function: A generator function that initializes and starts the pipeline.
    """
    def start_pipeline(*args, **kwargs):
        """
        Initializes and starts the pipeline.
        Args:
            *args: Positional arguments passed to the decorated function.
            **kwargs: Keyword arguments passed to the decorated function.
        Returns:
            generator: A generator instance initialized and ready to be started.
        """
        nf = f(*args, **kwargs)
        next(nf)
        return nf
    return start_pipeline

@pipeline
def ingest(corpus, targets):
    """
    Ingests text data into the pipeline.
    Args:
        corpus (iterable): A collection of text data to be ingested.
        targets (iterable): A collection of pipeline targets to receive the ingested data.
    """
    for text in corpus:
        for t in targets:
            t.send(text)
        yield

@pipeline
def tokenize_sentences(targets):
    while True:
        text = (yield)
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            for target in targets:
                target.send(sentence)

@pipeline
def tokenize_words(targets):
    """
    Tokenizes words using NLTK word_tokenize function.
    Args:
        targets (iterable): A collection of pipeline targets to receive the tokenized words.
    """
    while True:
        sentence = (yield)
        words = nltk.word_tokenize(sentence)
        for target in targets:
            target.send(words)

@pipeline
def pos_tagging(targets):
    """
    Performs part-of-speech tagging using NLTK pos_tag function.
    Args:
        targets (iterable): A collection of pipeline targets to receive the tagged words.
    """
    while True:
        words = (yield)
        tagged_words = nltk.pos_tag(words)
        
        for target in targets:
            target.send(tagged_words)

@pipeline
def ne_chunking(targets):
    """
    Performs named entity recognition using NLTK ne_chunk function.
    Args:
        targets (iterable): A collection of pipeline targets to receive the named entity tagged words.
    """
    while True:
        tagged_words = (yield)
        ner_tagged = nltk.ne_chunk(tagged_words)
        for target in targets:
            target.send(ner_tagged)

@pipeline
def printline(title):
    """
    Prints lines received by the pipeline with an optional title.
    Args:
        title (str): The title to be printed before each line.
    """
    while True:
        line = (yield)
        print(title)
        print(line)

