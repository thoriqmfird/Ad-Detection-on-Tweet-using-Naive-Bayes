import pandas as pd
import numpy as np
from preproc import Preproc
from training import Training
from testing import Testing

#Pre-Processing data
data_latih = pd.read_excel('DataLatih.xlsx') #4
preproc_latih = Preproc(data_latih["Text"])
hasil_normalisasi_latih = preproc_latih.data_normalized
hasil_preproc_latih = preproc_latih.preprocessing()
# print(hasil_preproc_latih)

data_uji = pd.read_excel('DataUji.xlsx') #1
preproc_uji = Preproc(data_uji["Text"])
hasil_normalisasi_uji = preproc_uji.data_normalized
hasil_preproc_uji = preproc_uji.preprocessing()

# Training
training = Training(data_latih['Class'], hasil_preproc_latih,hasil_normalisasi_latih)
kelas = np.unique(training.kategori)
prior = training.peluang_kelas()
##bow
likelihood_bow = training.peluang_fitur_bow()
##numeric & text
likelihood_text = training.peluang_fitur_text()
likelihood_num = training.peluang_fitur_num()

#testing
testing = Testing(kelas, hasil_normalisasi_uji, hasil_preproc_uji, prior, likelihood_bow, likelihood_text, likelihood_num)
hasil_klasifikasi = testing.naive_bayes()
hasil = pd.DataFrame(data_uji["Text"].values, columns=["Data Uji:"])
hasil["Hasil Klasifikasi:"] = hasil_klasifikasi
print(hasil)

#evaluasi
kelas_aktual = data_uji["Class"]

y_actu = pd.Series(kelas_aktual, name='Aktual')
y_pred = pd.Series(hasil_klasifikasi, name='Prediksi')
df_confusion = pd.crosstab(y_pred, y_actu)
print(df_confusion,"\n")

tp = df_confusion["Iklan"]["Iklan"]
fp = df_confusion["Iklan"]["Bukan"]
fn = df_confusion["Bukan"]["Iklan"]
tn = df_confusion["Bukan"]["Bukan"]

precision = tp/(fp+tp)
recall = tp/(fn+tp)
fmeasure = (2*precision*recall)/(precision+recall)
accuracy = (tp+tn)/(tp+tn+fp+fn)

print("\nPrecision: ", precision)
print("\nRecall: ", recall)
print("\nF-Measure: ", fmeasure)
print("\nAccuracy: ", accuracy)
