from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response

class PDFUploadView(APIView):
    parser_classes = (FileUploadParser,)

    def put(self, request, filename, format=None):
        print("askdjçfkalsd")
        file_obj = request.data['file']
        print(file_obj)
        # ...
        # do some stuff with uploaded file
        # ...
        return Response(status=204)
