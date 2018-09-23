import pickle
import os
import PyPDF2

from subprocess import call
from .pre_processing import CorpusHandler
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.shortcuts import render
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView

MY_MODEL = 'xgboost.pkl'
MODEL_PATH = os.path.join('../../models/', MY_MODEL)
PDF_TEMP = 'server-side.pdf'
TEXT_TEMP= 'temp.txt'

class ClassifierView(APIView):
    """
    Return the classification of PDF file.
    """

    parser_classes = (FileUploadParser,)

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
    
        return open('temp.txt', 'r').readlines()

    def _clean_temporary_files(self):
        os.remove(PDF_TEMP)
        os.remove(TEXT_TEMP)

    def put(self, request, filename, format=None):
        """
        Send to server the PDF file.
        """

        file_obj = request.data['file']
        
        path = default_storage.save('server-side.pdf', ContentFile(file_obj.read()))

        print(self._get_text('server-side.pdf'))

        #self._clean_temporary_files()

        # TODO change this accuracy
        return Response(status=200)
