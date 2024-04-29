import collections
import nltk
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import feature_extraction

def feature_extractor(tokens):
    '''Converts each word token into a feature.'''
    return dict(collections.Counter(tokens))

def feature_vectorizer(corpus):
    '''Preprocesses entire body of text data.'''
    sa_stop_words = nltk.corpus.stopwords.words('english')
    '''Create a list of exceptions, as these stopwords may change a sentence's sentiment if removed.'''
    sa_white_list = ['what', 'but', 'if', 'because', 'as', 'until', 'against', 'up', 'down', 'in', 'out',
                    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'why',
                    'how', 'all', 'any', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                    'same', 'so', 'than', 'too', 'can', 'will', 'just', 'don', 'should']
    '''Remove stop words except for those specified in the white list.'''
    sa_stop_words = [sw for sw in sa_stop_words if sw not in sa_white_list]
    '''Instantiate the vectorizer.'''
    count_vectorizer = feature_extraction.text.CountVectorizer(
        lowercase=True,
        tokenizer=nltk.word_tokenize,
        min_df=2,
        ngram_range=(1, 2),
        stop_words=sa_stop_words
    )
    '''Run the vectorizer on the body of text ('corpus').'''
    processed_corpus = count_vectorizer.fit_transform(corpus)
    processed_corpus = feature_extraction.text.TfidfTransformer().fit_transform(processed_corpus)
    return processed_corpus

def data_integrity_check(df, title='', include_non_numeric=True):
    '''Check for nulls, duplicates, etc and perform basic EDA.'''
    results = []
    for col in df:
        result = {
            'Column': col,
            'Null Values': df[col].isnull().sum(),
            'Duplicate Values': df[col].duplicated().sum(),
            'Data Type': df[col].dtype
        }
        if include_non_numeric or df[col].dtype in ['int64', 'float64']:
            result['Unique Values'] = df[col].nunique()
            if df[col].dtype in ['int64', 'float64']:
                result['Mean'] = df[col].mean()
                result['Median'] = df[col].median()
                result['Mode'] = stats.mode(df[col])
                result['Range'] = df[col].max() - df[col].min()
                result['Skew'] = df[col].skew()
                result['Kurtosis'] = df[col].kurtosis()
        if df[col].dtype == 'object':  
            result['Min Text Length'] = df[col].str.len().min()
            result['Max Text Length'] = df[col].str.len().max()
            '''Calculate mean and median text lengths'''
            text_lengths = df[col].str.len()
            result['Mean Text Length'] = np.mean(text_lengths)
            result['Median Text Length'] = np.median(text_lengths)
        results.append(result)
    result_df = pd.DataFrame(results)
    result_df['Source'] = title
    return result_df

def print_classification_report(y_true, y_pred):
    '''Get a classification report for performance metric inspection.'''
    report = classification_report(y_true, y_pred)
    print(report)