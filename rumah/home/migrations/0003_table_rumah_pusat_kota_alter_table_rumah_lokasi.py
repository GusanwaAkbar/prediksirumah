# Generated by Django 4.2.4 on 2023-11-13 08:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0002_alter_table_rumah_tanggal_sp_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='table_rumah',
            name='Pusat_kota',
            field=models.CharField(default='Pusat', max_length=50),
        ),
        migrations.AlterField(
            model_name='table_rumah',
            name='Lokasi',
            field=models.CharField(max_length=100),
        ),
    ]