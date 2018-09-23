from django.shortcuts import render, render_to_response
from django.http import HttpResponse
from django.template import Context, loader
from django.http import HttpResponseRedirect
from .forms import UploadFileForm

# Create your views here.
def index(request):
    return render(request, 'index.html')
def about(request):
    return render(request, 'about.html')

# Imaginary function to handle an uploaded file.


def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            return HttpResponseRedirect('/success/url/')
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})
