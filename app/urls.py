from django.urls import path
from .views import AIView,ProjectListView

urlpatterns = [
    path('chat/',AIView,name='ai-view'),
    path('csvData/',ProjectListView,name='csv-data'),
]
