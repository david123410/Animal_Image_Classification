# Animal Image Classification (Animals-10)

Proyek klasifikasi gambar untuk mengenali 10 jenis hewan menggunakan transfer learning dengan **MobileNetV2**. Model dilatih dan dievaluasi menggunakan dataset [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dari Kaggle.

## Deskripsi

Model mengklasifikasikan gambar ke dalam 10 kelas hewan berikut:

Label asli menggunakan bahasa italia yaitu
cane = anjing 
cavallo = kuda 
elefante = gajah 
farfalla = kupu-kupu 
gallina = ayam 
gatto = kucing 
mucca = sapi 
pecora = domba 
ragno = laba-laba 
scoiattolo = tupai 

Pendekatan yang digunakan:
- **Transfer learning** dengan backbone MobileNetV2 (bobot pretrained ImageNet)
- **Data augmentation** (flip, rotation, zoom, contrast, translation)
- **Two-stage training**: (1) freeze backbone, latih head klasifikasi, (2) fine-tuning sebagian layer backbone dengan learning rate kecil
- **Class weighting** untuk mengatasi distribusi kelas yang tidak seimbang
- Split dataset **train/val/test yang benar-benar terpisah** (menghindari data leakage)

## Hasil

- **Akurasi pada test set: 96.11%**
- Precision, recall, dan f1-score seluruh kelas berada di atas 0.90

```
              precision    recall  f1-score   support

        cane       0.96      0.97      0.97       730
     cavallo       0.96      0.90      0.93       394
    elefante       0.95      0.99      0.97       218
    farfalla       0.98      0.98      0.98       318
     gallina       0.97      0.98      0.97       466
       gatto       0.97      0.97      0.97       251
       mucca       0.90      0.91      0.91       281
      pecora       0.90      0.95      0.92       273
       ragno       1.00      0.98      0.99       724
  scoiattolo       0.95      0.96      0.96       280

    accuracy                           0.96      3935
   macro avg       0.96      0.96      0.96      3935
weighted avg       0.96      0.96      0.96      3935
```

## Struktur Repository

```
Animal_Image_Classification/
├── README.md
├── requirements.txt
├── Submission_Akhir_Perbaikan.ipynb   # notebook utama (training & evaluasi)
├── class_names.json                   # mapping index -> nama kelas
├── label.txt                          # daftar nama kelas (format teks)
└── models/
    ├── mobilenetv2_animals10.keras    # model format Keras
    └── mobilenetv2_animals10.tflite   # model format TensorFlow Lite
```

## Dataset

Dataset **tidak disertakan** dalam repository ini karena ukurannya besar (~26.000 gambar). Silakan unduh dari sumber aslinya:

🔗 [Animals-10 Dataset — Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

Setelah diunduh, ekstrak dataset sehingga strukturnya menjadi:
```
animals10/
├── cane/
├── cavallo/
├── elefante/
├── farfalla/
├── gallina/
├── gatto/
├── mucca/
├── pecora/
├── ragno/
└── scoiattolo/
```

## Cara Menjalankan

1. Clone repository ini:
   ```bash
   git clone https://github.com/david123410/Animal_Image_Classification.git
   cd Animal_Image_Classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Unduh dan letakkan dataset Animals-10 sesuai struktur folder di atas.

4. Buka dan jalankan `Submission_Akhir_Perbaikan.ipynb` (disarankan menggunakan Google Colab dengan GPU) secara berurutan dari cell awal hingga akhir.

## Menggunakan Model yang Sudah Dilatih

Untuk melakukan prediksi tanpa melatih ulang model, gunakan file `models/mobilenetv2_animals10.tflite`:

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json

IMG_SIZE = 224

with open("class_names.json") as f:
    class_names = json.load(f)

interpreter = tf.lite.Interpreter(model_path="models/mobilenetv2_animals10.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img).astype("float32")
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    pred_idx = int(np.argmax(output_data))
    confidence = float(output_data[0][pred_idx]) * 100
    return class_names[pred_idx], confidence

label, confidence = predict("contoh_gambar.jpg")
print(f"Prediksi: {label} ({confidence:.1f}%)")
```

## Teknologi yang Digunakan

- TensorFlow / Keras
- MobileNetV2 (transfer learning)
- scikit-learn (evaluasi metrik)
- Pillow, Matplotlib, Seaborn (visualisasi)
- split-folders (pembagian dataset)
