from django.shortcuts import render, get_object_or_404
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

import folium
import html

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma

from .models import Measurement
from .forms import MeasurementModelForm
from .utils import get_geo, get_center_coordinates, get_zoom, get_client_ip
from langchain_community.embeddings import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="./multilingual-e5-large")
COUNT = 3

# Create your views here.

def calculate_distance_view(request):
    form = MeasurementModelForm(request.POST or None)

    location_lat, location_lon = (55.751477, 37.619003)
    #initial folium map modification.    
    m = folium.Map(width=800, height=500, location=get_center_coordinates(location_lat, location_lon), zoom_start=10)

    if form.is_valid():
        instance = form.save(commit=False)
        ####
        db1 = Chroma(persist_directory="./db/", embedding_function=embedding_function, collection_name="reviews_collection_001")
        docs1 = db1.similarity_search_with_relevance_scores(form.cleaned_data.get('destination'), k=50)
        k=0
        names = []
        for d in docs1:
            if k == COUNT:
                break
            folium.Marker([d[0].metadata["lat"], d[0].metadata["lon"]], tooltip=d[0].metadata["name"], popup=d[0].page_content, icon=folium.Icon(color='red', icon='cloud')).add_to(m)
            if not d[0].metadata["name"] in names:
                names.append(d[0].metadata["name"])
                k+=1
        #####
        #Save to the database.
        #instance.save()

    #do an html representation of m
    m = m._repr_html_()


    context = {
        'distance': 0,
        'destination': 0,
        'form': form,
        'map':m,
    }

    return render(request, 'measurements/main.html', context)
