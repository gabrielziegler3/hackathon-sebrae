from django.urls import path
from .views import ClassifierView
from django.conf.urls import url

urlpatterns = [
            url(r'upload/(?P<filename>[^/]+)$', ClassifierView.as_view()),
]
