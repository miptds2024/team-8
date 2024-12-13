from django.urls import path
from .views import process_query

app_name = 'measurements'

urlpatterns = [
    path('', process_query, name='calculate_view'),
]
