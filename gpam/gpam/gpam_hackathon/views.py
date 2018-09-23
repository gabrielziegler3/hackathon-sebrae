from django.shortcuts import render
from rest_framework.decorators import api_view

#@api_view(['POST']):
#def get_pdf_classification(request):
#    """
#    Return classification for this pdf in request
#    """
#
#    if request.method == 'POST':
#
#        print(request.data)
#
#        data_text = get_text_information(request.data)
#        model_classification =  get_class(data_text)
#
#        return model_classification
def index(request):
    return render(request, 'gpam_hackathon/index.html')
