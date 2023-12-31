"""
URL configuration for rumah project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from home import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.main_map),
    path('map', views.main_map),

    path('run1', views.run1),
    path('run2', views.run2),
    path('run3', views.run3),
    path('run4', views.run4),
    path('run5', views.run5),

    #path('predict', views.predict),

    path('simulasi', views.simulasi_list, name='simulasi_list'),
    path('add_simulasi/', views.add_simulasi, name='add_simulasi'),

   
]
