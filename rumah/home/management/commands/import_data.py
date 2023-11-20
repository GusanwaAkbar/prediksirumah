import csv
from datetime import datetime
from django.core.management.base import BaseCommand
from home.models import Table_rumah

class Command(BaseCommand):
    help = 'Import data from CSV into the Table_rumah model'

    def handle(self, *args, **options):
        csv_file_path = '/home/gusanwa/AA_Programming/huda/data4.csv'  # Update this with the actual path to your CSV file

        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # Assuming your CSV has columns id, Bulan, No., No_laporan, Tanggal_laporan, and other fields...
                Table_rumah.objects.create(
                    id=int(row['id']),
                    Bulan=row['Bulan'],
                    No=row['No.'],
                    No_laporan=row['No_laporan'],
                    Tanggal_laporan=row['Tanggal_laporan'],
                    Tanggal_penilaian=row['Tanggal_penilaian'],
                    NO_SP=row['NO_SP'],
                    Tanggal_SP=row['Tanggal_SP'],
                    Nama_debitur=row['Nama_debitur'],
                    Nama_Pemberi_Tugas=row['Nama_Pemberi_Tugas'],
                    Tujuan=row['Tujuan'],
                    Obyek=row['Obyek'],
                    Lokasi=row['Lokasi'],
                    Kota_Kabupaten=row['Kota/Kabupaten'],
                    LT=int(row['LT']),
                    LB=int(row['LB']),
                    CP=row['CP'],
                    Supervisi=row['Supervisi'],
                    Penilai=row['Penilai'],
                    Pelaksana=row['Pelaksana'],
                    Fee_total=int(row['Fee_total']),
                    dpp=int(row['dpp']),
                    ppn=int(row['ppn']),
                    Indikasi_nilai=row['Indikasi_nilai'],
                    Kesimpulan_nilai=row['Kesimpulan_nilai'],
                    Report=row['Report'],
                    Titik_koordinat=row['Titik_koordinat'],
                    Nama=row['Nama'],
                    Jumlah_kamar=int(row['Jumlah_kamar']),
                    Jumlah_kamarmandi=int(row['Jumlah_kamarmandi']),
                    Jumlah_lantai=int(row['Jumlah_lantai']),
                    Pusat_kota=row['Pusat_kota']
                )

        self.stdout.write(self.style.SUCCESS('Data imported successfully'))
