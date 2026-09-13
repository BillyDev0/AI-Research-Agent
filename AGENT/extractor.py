from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import json
from logger import logger
from pydantic import BaseModel
from typing import List

model = ChatOllama(
    model="qwen2.5:7b",
    top_p=0.2,
    temperature=0.3,
    num_thread=8,      # sesuaikan jumlah core CPU kamu
    num_ctx=2048,      # jangan set ctx lebih besar dari yang perlu — makin besar makin lambat
)

class DataItem(BaseModel):
    label: str
    value: str

class Entitas(BaseModel):
    entitas: str
    data: List[DataItem]

class ExtractorOutput(BaseModel):
    hasil: List[Entitas]


structured_output=model.with_structured_output(ExtractorOutput, include_raw=True)
template = """
# ROLE
Kamu adalah Extractor Agent. Tugasmu mengambil DATA_MENTAH (hasil model sebelumnya), merangkum, dan menyaring HANYA data yang relevan dan dibutuhkan GOAL_USER. Kamu TIDAK boleh menambah, mengarang, atau menyimpulkan data yang tidak ada di DATA_MENTAH.

GOAL_USER:
{goal_user}

DATA_MENTAH:
{data_mentah}

# ATURAN JUMLAH (WAJIB, PALING PENTING)
1. Jika GOAL_USER menyebutkan jumlah spesifik (misal "5 gunung", "3 HP", "top 10", "1 saham"), OUTPUT WAJIB berisi TEPAT sejumlah itu — TIDAK LEBIH, TIDAK KURANG — meski DATA_MENTAH menyediakan data lebih banyak dari itu.
2. Jika data di DATA_MENTAH lebih banyak dari yang diminta, pilih HANYA sejumlah yang diminta sesuai konteks goal (misal jika diminta "tertinggi", ambil yang nilainya tertinggi; jika diminta "termurah", ambil yang termurah, dst).
3. Sebelum menuliskan output final, HITUNG ULANG jumlah entitas yang sudah kamu susun — cocokkan dengan angka yang diminta GOAL_USER. Jika tidak sama, tambah atau buang entitas sampai jumlahnya PERSIS sesuai permintaan.
4. Jika GOAL_USER tidak menyebutkan jumlah spesifik, tampilkan semua entitas yang relevan dan ditemukan di DATA_MENTAH.

# ATURAN DATA
5. Ambil HANYA data yang relevan menjawab GOAL_USER, buang sisanya.
6. DILARANG menambah/mengarang informasi yang tidak ada di DATA_MENTAH.
7. Setiap entitas hanya berisi data yang relevan dengan yang diminta GOAL_USER (contoh: user minta harga & chipset → jangan sertakan atribut lain yang tidak diminta).
8. Jika satu jenis data punya BEBERAPA VARIAN/PECAHAN (misal harga per kapasitas penyimpanan, harga per warna, ketinggian per titik puncak, dll), JANGAN digabung jadi satu nilai — pecah masing-masing varian jadi baris data terpisah dengan label yang jelas menyebutkan variannya (contoh: "harga varian 128GB", "harga varian 256GB").
9. Sertakan sumber jika ada di DATA_MENTAH.
10. Jika suatu data yang diminta GOAL_USER tidak ditemukan di DATA_MENTAH, isi nilainya dengan "tidak ditemukan" — jangan dikosongkan atau dikarang.
11. Jika DATA_MENTAH berupa penjelasan/konsep (bukan produk dengan atribut fisik), isi "entitas" dengan nama topik/konsep yang dibahas, dan pecah isi penjelasan menjadi beberapa "label"-"value" berdasarkan poin-poin kunci (aturan, definisi, sebab-akibat, kriteria, dll) — JANGAN menyalin satu paragraf penuh jadi satu "value" panjang, pecah jadi unit informasi yang lebih kecil dan jelas.

# FORMAT OUTPUT (WAJIB, JSON, TANPA TEKS LAIN)
Setiap entitas berisi list data berbentuk {{"label": "...", "value": "..."}} — bebas jumlahnya sesuai berapa banyak data/varian yang relevan ditemukan untuk entitas itu.

[
    {{
        "entitas": "...",
        "data": [
            {{"label": "...", "value": "..."}}
        ]
    }}
]

# CONTOH 1 (jangan digunakan seagai output)
GOAL_USER: "Cari harga dan chipset iPhone 14"
Output:
[
  {{
    "entitas": "iPhone 14",
    "data": [
      {{"label": "harga", "value": "Rp 1"}},
      {{"label": "chipset", "value": "A15 Bionic"}},
      {{"label": "sumber", "value": "tokopedia"}}
    ]
  }}
]


"""

def extractor(goal_user,result_evaluator,):
    prompt=ChatPromptTemplate.from_template(template)
    chain=prompt|structured_output
    result=chain.invoke({"goal_user":goal_user,"data_mentah":result_evaluator})
    logger.info(f"USAGE TOKEN EXTRACTOR: {result["raw"].response_metadata}")
    return result

