import pickle
import os

from subprocess import call
from .pre_processing import CorpusHandler, BatchProcessing
from sklearn.feature_extraction.text import TfidfVectorizer

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.shortcuts import render
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView


SITE_ROOT = os.path.dirname(os.path.realpath(__file__))
PDF_TEMP = 'server-side.pdf'
TEXT_TEMP = 'temp.txt'
MY_MODEL = 'models/logres.pkl'
TFIDF = 'models/tfidf.pkl'

MODEL_TEMP = TEXT_TEMP
MODEL_PATH = os.path.join(SITE_ROOT, MY_MODEL)
TFIDF_PATH = os.path.join(SITE_ROOT, TFIDF)

print(SITE_ROOT)
class ClassifierView(APIView):
    """
    Return the classification of PDF file.
    """

    parser_classes = (FileUploadParser,)

    def _load_tfidf(self):
        """
        Return TFIDF
        """
        tfidf = None

        with open(TFIDF_PATH, 'rb') as tfidf_file:
            tfidf = pickle.load(tfidf_file)

        return tfidf

    def _load_model(self):
        """
        Return de Machine Learning Model
        """
        model_loaded = None

        with open(MODEL_PATH, 'rb') as model_file:
            model_loaded = pickle.load(model_file)

        return model_loaded

    def _get_text(self, pdf_file_path):
        """
        Get all text from a PDF
        """
        call('pdftotext server-side.pdf temp.txt', shell=True)

        return open(MODEL_TEMP, 'r').readlines()

    def _clean_temporary_files(self):
        os.remove(PDF_TEMP)
        os.remove(MODEL_TEMP)

    def pre_process_text(self, corpus):
        pipe = [str.lower, 'clean_email', 'clean_site', 'clean_document',
                'transform_token', 'remove_letter_number', 'clean_number',
                'clean_spaces', 'clean_alphachars','remove_stop_words']

        processed_corpus = BatchProcessing.parallel_processing(pipe, corpus, progress=True)

        return processed_corpus

    def post(self, request, filename, format=None):
        """
        Send to server the PDF file.
        """

        print("========This is debug: ", request.data['file'])  
        # Load model and tfidf
        tfidf = self._load_tfidf()
        model = self._load_model()

        file_obj = request.data['file']

        path = default_storage.save('server-side.pdf', ContentFile(file_obj.read()))

        # Read PDF text
        pdf_text = self._get_text('server-side.pdf')
        print(self._get_text('server-side.pdf'))

        # Process text data
        processed_pdf = self.pre_process_text(pdf_text)

        # Transform text to tfidf scores
        tfidf_corpus = tfidf.transform(processed_pdf)

        # Predict probability to a given text
        prediction = model.predict_proba(tfidf_corpus)
        print('O processo encaminhado tem {:.2f}% de chance de ser admitido para julgamento no STF'.format(prediction))

        #self._clean_temporary_files()

        # TODO change this accuracy
        print(request.data)
    
        return render(request, 'index.html', {'flag': 0})
