# Generated by Django 4.1.7 on 2023-06-30 13:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backendapi', '0002_userauthority_is_receptionist'),
    ]

    operations = [
        migrations.AddField(
            model_name='userauthority',
            name='Gender',
            field=models.CharField(default=1, max_length=50),
            preserve_default=False,
        ),
    ]
