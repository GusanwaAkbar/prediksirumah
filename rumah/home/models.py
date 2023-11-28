from django.db import models



class Table_rumah(models.Model):
    Bulan = models.CharField(max_length=50)
    No = models.IntegerField()
    No_laporan = models.CharField(max_length=50)
    Tanggal_laporan = models.CharField(max_length=50)
    Tanggal_penilaian =  models.CharField(max_length=50)
    NO_SP = models.CharField(max_length=50)
    Tanggal_SP =  models.CharField(max_length=50)
    Nama_debitur = models.CharField(max_length=100)
    Nama_Pemberi_Tugas = models.CharField(max_length=100)
    Tujuan = models.CharField(max_length=100)
    Obyek = models.CharField(max_length=100)
    Lokasi = models.CharField(max_length=300)
    Kota_Kabupaten = models.CharField(max_length=100)
    LT = models.IntegerField()
    LB = models.IntegerField()
    CP = models.CharField(max_length=200)
    Supervisi = models.CharField(max_length=100)
    Penilai = models.CharField(max_length=100)
    Pelaksana = models.CharField(max_length=100)
    Fee_total = models.IntegerField()
    dpp = models.IntegerField()
    ppn = models.IntegerField()
    Indikasi_nilai = models.IntegerField()
    Kesimpulan_nilai = models.CharField(max_length=100)
    Report = models.CharField(max_length=100)
    Titik_koordinat = models.CharField(max_length=50)
    Nama = models.CharField(max_length=100)
    Jumlah_kamar = models.IntegerField()
    Jumlah_kamarmandi = models.IntegerField()
    Jumlah_lantai = models.IntegerField()
    Pusat_kota = models.CharField(max_length=50, default="Pusat")


class Simulasi(models.Model):
    Nama = models.CharField(max_length=255)
    Titik_koordinat = models.CharField(max_length=100)
    Kota_Kabupaten = models.CharField(max_length=100)
    LT = models.IntegerField()
    LB = models.IntegerField()
    Tujuan = models.CharField(max_length=255)
    Obyek = models.CharField(max_length=255)
    Indikasi_nilai = models.IntegerField()
    Jumlah_kamar = models.IntegerField()
    Jumlah_kamarmandi = models.IntegerField()
    Jumlah_lantai = models.IntegerField()
    Pusat_kota = models.CharField(max_length=50)
    Prediksi = models.FloatField(max_length=255)

    def __str__(self):
        return self.Nama
    

