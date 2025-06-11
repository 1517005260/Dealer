"""dealer_demo URL配置

urlpatterns列表将URL路由到视图。更多信息请参见：
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
示例：
函数视图
    1. 添加导入：from my_app import views
    2. 添加URL到urlpatterns：path('', views.home, name='home')
基于类的视图
    1. 添加导入：from other_app.views import Home
    2. 添加URL到urlpatterns：path('', Home.as_view(), name='home')
包含另一个URLconf
    1. 导入include()函数：from django.urls import include, path
    2. 添加URL到urlpatterns：path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from dealer import views as app_views

urlpatterns = [
    path('cancer/all/', app_views.query_cancer),
    path('cancer/id', app_views.query_cancer_by_id),
    path('chess/all/', app_views.query_chess),
    path('chess/id', app_views.query_chess_by_id),
    path('admin/', admin.site.urls),
    path('amp', app_views.query_amp),
    path('amp_shapley', app_views.query_amp_shapley),
    path('shapley', app_views.query_compensation),
    path('write_survey', app_views.write_survey),
    path('model/all', app_views.query_all_model),
    path('model/exp', app_views.query_limited_model),
    path('model/release', app_views.release_model),
    path('iris/all', app_views.query_iris),
    path('delete_model', app_views.delete_all_model)
]
