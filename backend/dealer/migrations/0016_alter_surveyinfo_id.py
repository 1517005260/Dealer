# Generated by Django 4.2.23 on 2025-06-11 12:58

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("dealer", "0015_remove_shapleyinfo_compensation"),
    ]

    operations = [
        migrations.AlterField(
            model_name="surveyinfo",
            name="id",
            field=models.BigAutoField(
                auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
            ),
        ),
    ]
