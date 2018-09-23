import pickle

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.shortcuts import render
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView

MY_MODEL = 'xgboost.pkl'

class ClassifierView(APIView):
    """
    Return the classification of PDF file.
    """
    parser_classes = (FileUploadParser,)

    def _load_model():
        pass

    def put(self, request, filename, format=None):

        file_obj = request.data['file']
        
        path = default_storage.save('server-side.pdf', ContentFile(file_obj.read()))

        return Response(status=200)
