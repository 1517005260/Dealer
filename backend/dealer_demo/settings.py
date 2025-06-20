"""
Django配置文件 - dealer_demo项目

由'django-admin startproject'生成，使用Django 3.1.7版本

关于此文件的更多信息，请参见：
https://docs.djangoproject.com/en/3.1/topics/settings/

设置及其值的完整列表，请参见：
https://docs.djangoproject.com/en/3.1/ref/settings/
"""
import os
from pathlib import Path

# 构建项目内部路径，如：BASE_DIR / 'subdir'
BASE_DIR = Path(__file__).resolve().parent.parent

# 快速启动开发设置 - 不适用于生产环境
# 参见：https://docs.djangoproject.com/en/3.1/howto/deployment/checklist/

# 安全警告：保持生产环境中使用的密钥秘密！
SECRET_KEY = 'q&vs4m*uxkm)*w^*9oz=^ruhjf+76=87k3!otz97_o=3gw9fbm'

# 安全警告：不要在生产环境中开启调试模式！
DEBUG = False
STATIC_ROOT = os.path.join(BASE_DIR, 'statics')

ALLOWED_HOSTS = ['*']

# 应用程序定义

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'dealer',
    'corsheaders',
    'import_export'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'dealer_demo.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'dealer_demo.wsgi.application'

# 数据库配置
# https://docs.djangoproject.com/en/3.1/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# 密码验证
# https://docs.djangoproject.com/en/3.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# 国际化
# https://docs.djangoproject.com/en/3.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True

# 静态文件配置 (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.1/howto/static-files/

STATIC_URL = '/static/'

CORS_ALLOW_CREDENTIALS = True
CORS_ORIGIN_ALLOW_ALL = True
# CORS_ORIGIN_WHITELIST = ('http://127.0.0.1:8080',
#                          'http://localhost:8080',)

CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]

CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
