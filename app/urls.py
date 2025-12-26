from django.urls import path
from .views import AIView

urlpatterns = [
    path('chat/',AIView,name='ai-view'),
]
