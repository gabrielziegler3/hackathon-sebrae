"""
Natural Language Processing module
"""
import re
import spacy
from spacy.lang import pt
import nltk
import multiprocessing
from nltk.stem.snowball import SnowballStemmer
from joblib import Parallel, delayed
from tqdm import tqdm


try:
    nltk.word_tokenize('some word')
except:
    nltk.download('punkt')

try:
    nltk.corpus.stopwords.words('portuguese')
except:
    nltk.download('stopwords')
finally:
    STOP_WORDS = pt.STOP_WORDS.union(
        set(nltk.corpus.stopwords.words('portuguese'))
    )


class CorpusHandler:
    """
    Handles text corpus
    """

    def __init__(self):
        pass

    # DEPRECATED
    @classmethod
    def clean_corpus(cls, corpus, **kwargs):
        """Apply regex in data to clean it"""

        pipe = [ 'lower', 'clean_email', 'clean_site', 'clean_document', 
                 'transform_token', 'remove_letter_number', 'clean_number', 
                 'remove_small_big_words', 'clean_spaces', ]
        return BatchProcessing.parallel_processing(pipe, corpus=corpus)

    @staticmethod
    def clean_number(document, **kwargs):
        """
        Use regex to remove numbers of text


        Arguments:
            :document: the document text to apply regex


        Returns:
            :document: the same document with numbers filtered


        Example:

        >>> CorpusHandler.clean_number("This is a 33")
        "This is a "
        """
        return re.sub(r'\s\d+\s', ' ', document)

    @staticmethod
    def clean_email(document, **kwargs):
        """
        Use regex to transform email in a token EMAIL.
        The composition of e-mail is local_part@domain


        Arguments:
            :document: the document text to apply regex


        Returns:
            :document: the same document with e-mail transformed

        References:
            https://tools.ietf.org/html/rfc5322#section-3.4.1


        Example:

        >>> CorpusHandler.clean_email("This is a mail@mail.com of example")
        "This is a EMAIL of example"
        """
        local_part = r"[0-9a-zA-Z!#$%&'*+-/=?^_`{|}~.]+"
        domain = r"[a-zA-Z][a-zA-Z0-9-.]*[a-zA-Z]\.\w{2,4}"
        document = re.sub(r"\s{}@{}".format(local_part, domain), ' EMAIL ', document)
        return document

    @staticmethod
    def clean_site(document, **kwargs):
        """
        Addaption for rules of https://tools.ietf.org/html/rfc3986#section-3 to
        stf's documents

        Arguments:
            :document: the document text to apply regex

        Returns:
            :document: the same document with site transformed

        Example:

        >>> CorpusHandler.clean_site("This is a http://example.com or http://192.168.0.255")
        "This is a SITE or SITE"
        """
        scheme = r"[a-zA-Z][a-zA-Z0-9+-.]*:?//?"
        host_ip = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        host_name = r"[a-zA-Z]+(\.[a-zA-Z0-9-_~]+)+"
        www = r'www\.{}'.format(host_name)
        scheme_host = r'({}{}|{}{}|{}{}|{})'.format(
            scheme, host_ip, scheme, www, scheme, host_name, www
        )
        port = r"(:\d+)?"
        resource = r"(/[a-zA-Z0-9-._~!$&'/*+,;=]*)?"
        query = r"(\?[a-zA-Z0-9-._~!$&'/*+,;=]*)?"
        fragment = r"(#[a-zA-Z0-9-._~!$&'/*+,;=]*)?"
        document = re.sub(r"{}{}{}{}{}".format(scheme_host, port, resource,
                                             query, fragment), ' SITE ', document)
        return document

    @staticmethod
    def transform_token(document, **kwargs):
        """
        Create custom tokens for laws and articles

        Arguments:
            :document: the document text to apply regex

        Returns:
            :document: the same document with laws and decrets

        Example:

        >>> CorpusHandler.transform_token("This is a Lei n.º 13105")
        "This is a LEI_13105"
        """
        word_number = r'(nº.?|número|numero|n.?|no.?)?'
        # Law number: 00/YEAR or 00
        number_law = r'([0-9]+)((\s|\.)+\d+)?'
        matchs = [('LEI', 'lei'), ('ARTIGO', r'art\.|art\w*'),
                  ('DECRETO', 'decreto'),]

        for word, regex in matchs:
            document = re.sub(r'{}\s*{}\s*{}'.format(regex, word_number, number_law),
                            r'{}_\2'.format(word), document, flags=re.I)

        return document

    @staticmethod
    def remove_small_big_words(document, **kwargs):
        """
        Remove words

        Arguments:
            :document: the document text to apply regex

        Returns:
            :document: the same document without small tokens

        Example:

        >>> CorpusHandler.transform_token("This is a text example")
        "This     text example"
        """
        # remove 2 chars
        document = re.sub(r'\s\w{0,2}\s', ' ', document)
        # remove bigger words ex.: infrainconstitucionalidade
        document = re.sub(r'\s\w{30,}\s', ' ', document)
        return document

    @staticmethod
    def remove_letter_number(document, **kwargs):
        """
        Remove any word with number but keep the preceded by _ for tokens

        Arguments:
            :document: the document text to apply regex

        Returns:
            :document: the same document without numbers and letters in same word

        Example:

        >>> CorpusHandler.remove_letter_number("This is a text ex4kample")
        "This is a text "
        """
        # Remove wor00 00wo 00wor00 wo00rd and keep WORD_000
        return re.sub(r'([A-Z]+_\d+)|[^ ]*\d+[^ ]*', r'\1', document)


    @staticmethod
    def clean_document(document, **kwargs):
        """
        Removes unwanted words from text

        Arguments:
            :document: the document text to apply regex

        Returns:
            :document: the same document without special chars

        Example:

        >>> CorpusHandler.clean_document("This »“ð example")
        "This      example"
        """

        # Replace 0.0 for 00
        document = re.sub(r'(\d)\.(\d)', r'\1\2', document)

        # Remove all non alphanumeric
        document = re.sub(r'\W', ' ', document)

        return document

    @staticmethod
    def clean_spaces(document, **kwargs):
        """
        Remove multiple spaces

        Arguments:
            :document: the document text to apply regex

        Returns:
            :document: the same document without spaces

        Example:

        >>> CorpusHandler.clean_spaces("This  is  a   example")
        "This is a example"
        """
        # Remove multiple spaces
        document = re.sub(r'\s+', ' ', document)
        document = document.strip()
        return document

    @staticmethod
    def clean_alphachars(document, **kwargs):
        """
        Remove any special letters

        Arguments:
            :document: the document text to apply regex

        Returns:
            :document: the same document with only alphachars

        Example:

        >>> CorpusHandler.clean_alphachars("This is a €³~ éxàmplê")
        "This is a    éxàmplê"
        """
        return re.sub(r'[^ ]*[^-_úíóõôéêãáàâa-zA-Z0-9ç ]+[^ ]*',
                    ' ', document)

    def clean_special_chars(document, **kwargs):
        """
        Replace special chars with ascii chars then remove words with special
        chars

        Arguments:
            :document: the document text to apply regex

        Returns:
            :document: the same document without accentuation

        Example:

        >>> CorpusHandler.clean_alphachars("This is a éxàmplê")
        "This is a example"
        """
        a_doc = re.sub(r'[ãáàâ]', 'a', document)
        e_doc = re.sub(r'[éê]', 'e', a_doc)
        i_doc = re.sub(r'[í]', 'i', e_doc)
        o_doc = re.sub(r'[óõô]', 'o', i_doc)
        u_doc = re.sub(r'[ú]', 'u', o_doc)

        document = re.sub(r'[^ ]*[^a-zA-Z0-9ç ]+[^ ]*', ' ', u_doc)
        return document

    nlp = spacy.load("pt_core_news_sm")
    @classmethod
    def lemmatize(cls, document, **kwargs):
        """

        Arguments:
            :document: the document text to lemmatize

        Returns:
            text_tokens (list): a list of tokens lemmatized

        Example:

        >>> CorpusHandler.lemmatize("This is a example")
        "This be a example"
        """

        texts = []
        # for token in document:
        doc = cls.nlp(document, disable=['parser', 'ner'])

        for token in doc:
            #print(token)
            texts.append(token.lemma_)

        return texts

    @staticmethod
    def tokenize(document, **kwargs):
        """
        Transform string documents in array of tokens.
        The default behavior is split in spaces

        Arguments:
            :document: the document text to extract tokens

        Returns:
            text_tokens (list): a list of tokens

        Example:

        >>> CorpusHandler.tokenize("This is a example")
        ["This", "is", "a", "example"]
        >>> CorpusHandler.tokenize(["This", "is", "a", "example"])
        ["This", "is", "a", "example"]
        """
        if isinstance(document, str):
            document = document.split()
        return document

    @classmethod
    def remove_stop_words(cls, document, stop_words=[], extra_stop_words=[], **kwargs):
        """
        Remove the stop words

        Arguments:
            :document: the document text remove stop words

        Returns:
            :document: a document without stop words

        Example:

        >>> CorpusHandler.remove_stop_words("Document to example method")
        "Document example method"
        >>> CorpusHandler.remove_stop_words("Document to example method", stop_words=['to', 'example'])
        "Document method"
        >>> CorpusHandler.remove_stop_words("Document to example method", extra_stop_words=['Document'] )
        "example method"
        >>> CorpusHandler.remove_stop_words("Document to example method", stop_words=['example'], extra_stop_words=['Document'])
        "to method"
        """

        tokens = cls.tokenize(document)

        words = set(stop_words) or STOP_WORDS

        if extra_stop_words != []:
            words = words.union(set(extra_stop_words))

        document = " ".join(filter(lambda x: x not in words, tokens))
        return document

    stemmer = SnowballStemmer("portuguese")
    @classmethod
    def snowball_stemmer(cls, document, **kwargs):
        """
        Use nltk Snowball Stemmer to stemmize words

        Arguments:
            :document: the document text remove stop words

        Returns:
            :document: a document with stemm

        Example:

        >>> CorpusHandler.snowball_stemmer("Exemplify the documents")
        "exemp the document"
        """
        tokens = cls.tokenize(document)
        document = ' '.join([cls.stemmer.stem(word) for word in tokens])
        return document


class BatchProcessing:
    """
    Execute a block of function in a corpus of text
    """

    base_class = CorpusHandler

    @classmethod
    def batch_processing(cls, functions, document, extra_kwargs={}):
        """
        Receive a list of functions to apply in a document of text

        Arguments:
            :functions: a list of functions to apply in document. It can be a str, function or dictionary
            :document: a text to apply transformations
            :extra_kwargs: a dictionary with function name and the kwargs

        Returns:
            :document: with all functions transformations

        Example:

        >>> BatchProcessing.batch_processing([str.lower], "EXEMPLIFY THE DOCUMENTS")
        "exemplify the documents"
        >>> BatchProcessing.batch_processing([int], "12")
        12
        >>> BatchProcessing.batch_processing([{int: {'base': 2}}], "10")
        2
        >>> BatchProcessing.batch_processing([int], "10", extra_kwargs={'int': {'base': 2}})
        2
        """
        for function in functions:
            kwargs={}
            # Extract the function or function name from dict item
            if isinstance(function, dict):
                if len(function) == 1:
                    function, kwargs = tuple(function.items())[0]
                else:
                    raise TypeError('Should be passed only a function per'
                        'position to functions')
            # Obtain a function from str value
            if isinstance(function, str):
                if hasattr(cls.base_class, function):
                    function = getattr(cls.base_class, function)
                else:
                    raise AttributeError('Function {}'.format(function) +
                                         'not implemented in {}'.format(cls.base_class))
            if function.__name__ in extra_kwargs:
                kwargs.update(extra_kwargs[function.__name__])
            document = cls.call_function(function, document, **kwargs)
        return document

    @staticmethod
    def call_function(function, document, **kwargs):
        """
        Execute a function with kwargs or just with document.

        Arguments:
            :function: function object to apply in document
            :document: a text

        Returns:
            :document: with function transformation

        Example:

        >>> BatchProcessing.call_function(str.lower, "EXEMPLIFY THE DOCUMENTS")
        "exemplify the documents"
        >>> BatchProcessing.call_function(int, "12")
        12
        >>> BatchProcessing.call_function(int, "10", base=2)
        2
        """
        if callable(function):
            if kwargs != {}:
                return function(document, **kwargs)
            else:
                return function(document)
        else:
            raise TypeError('{} object is not callable'.format(function))

    @classmethod
    def parallel_processing(cls, functions, corpus=[], n_jobs=-1,
            progress=False, extra_kwargs={}):
        """
        Apply a function in corpus using parallel lib

        Arguments:
            :functions: a list of functions to apply in document. It can be a str, function or dictionary
            :corpus: a array of texts to apply a function
            :n_jobs: the number of parallel executions, if not pass, it will use all cores
            :progress: show the status of processing, by default use tqdm
            :extra_kwargs: a dictionary with function name and the kwargs

        Returns:
            :data: a list of texts with all functions transformations

        Example:

        >>> BatchProcessing.parallel_processing([str.lower], ["EXEMPLIFY THE DOCUMENTS", "TO LOWER"])
        ["exemplify the documents", "to lower"]
        >>> BatchProcessing.parallel_processing([{int: {'base': 2}}], ["10", "100", "101"])
        [2, 4, 5]
        >>> BatchProcessing.parallel_processing([int], ["10", "100", "101"], extra_kwargs={'int': {'base': 2}})
        [2, 4, 5]
        >>> BatchProcessing.parallel_processing([int], ["10", "100", "101", "110", "111"], extra_kwargs={'int': {'base': 2}}, progress=True)
        100%|█████████████████████████████████████████████████| 5/5 [00:00<00:00, 84.99it/s]
        [2, 4, 5, 6, 7]
        """
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        parallel = range(0)
        # Conditional use of status
        if progress:
            corpus_len = len(corpus)
            parallel = tqdm(
                            (delayed(cls.batch_processing)(functions, document,
                                extra_kwargs)
                            for document in corpus),
                            total=corpus_len
                    )
        else:
            parallel = (delayed(cls.batch_processing)(functions, document,
                extra_kwargs)
                            for document in corpus)

        data = Parallel(n_jobs=n_jobs)(parallel)
        return data
