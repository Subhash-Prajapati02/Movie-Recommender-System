from django.urls import path
from .views import *

urlpatterns = [
    path('', Home.as_view(), name="home"),
    path('movie/<str:movie>/', MovieDetail.as_view(), name="movie_detail"),
]
