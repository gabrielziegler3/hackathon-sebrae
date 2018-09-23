from django import forms

class UploadFileForm(forms.Form):

    filer = forms.FileField()

