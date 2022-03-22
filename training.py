import numpy as np
import pandas as pd
import re


class Training(object):
    def __init__(self, kategori, data, data_norm):
        self.kategori = kategori
        self.list_kelas = np.unique(self.kategori)
        self.hasil_preproc_latih = data
        self.data_norm = data_norm
        self.term = np.unique(sum(self.hasil_preproc_latih, []))
        self.kolom = [i + 1 for i in range(len(self.hasil_preproc_latih))]
        self.hasil_bow_tf = self.bow_tf()

    def bow_tf(self):
        data_latih = self.hasil_preproc_latih
        kolom = [[i + 1 for i in range(len(data_latih))]]
        term = np.unique(sum(data_latih, []))
        hasil_raw_tf = pd.DataFrame(0, index=term, columns=kolom)
        for i, doc in enumerate(data_latih):
            for term in doc:
                hasil_raw_tf.loc[term, i+1] += 1
        # print(hasil_raw_tf)
        return hasil_raw_tf

    def peluang_kelas(self):
        term_group = self.kategori.groupby(self.kategori)
        kelas = {}
        for i in np.unique(self.kategori):
            temp = term_group.get_group(i).index.tolist()
            kelas[i] = [j + 1 for j in temp]
        peluang_kelas = {}
        for i in kelas:
            peluang_kelas[i] = len(kelas[i]) / len(self.hasil_preproc_latih)
        return peluang_kelas

    def peluang_fitur_bow(self):
        term_group = self.kategori.groupby(self.kategori)
        kelas = {}
        for i in np.unique(self.kategori):
            temp = term_group.get_group(i).index.tolist()
            kelas[i] = [j + 1 for j in temp]
        total_term = []
        for i in self.hasil_bow_tf:
            jumlah = sum(self.hasil_bow_tf[i])
            total_term.append(jumlah)
        
        total_iklan = 0
        index_iklan = kelas["Iklan"]
        for i in index_iklan:
            total_iklan += total_term[i-1]
        # print(total_iklan)

        total_bukan = 0
        index_bukan = kelas["Bukan"]
        for i in index_bukan:
            total_bukan += total_term[i-1]
        # print(total_bukan)

        peluang_term_bow = pd.DataFrame(0, index=self.hasil_bow_tf.index, columns=["Iklan"])
        peluang_term_bow["Iklan"] = (self.hasil_bow_tf[index_iklan].sum(axis=1) + 1)/(total_iklan + len(self.hasil_bow_tf))
        peluang_term_bow["Bukan"] = (self.hasil_bow_tf[index_bukan].sum(axis=1) + 1)/(total_bukan + len(self.hasil_bow_tf))
        print(peluang_term_bow,"\n")

        dict_peluang_term_bow = {}
        
        for i in range(len(self.list_kelas)):
            for j in self.list_kelas:
                key = peluang_term_bow.index + '|' + j
                val = peluang_term_bow[j]
                merg = dict(zip(key, val))
                dict_peluang_term_bow.update(merg)
        
        nilai_default_iklan = 1/(total_iklan + len(self.hasil_bow_tf))
        nilai_default_bukan = 1/(total_bukan + len(self.hasil_bow_tf))
        dict_peluang_term_bow.update({"default|Iklan" : nilai_default_iklan})
        dict_peluang_term_bow.update({"default|Bukan" : nilai_default_bukan})
        
        return dict_peluang_term_bow

    def peluang_fitur_num(self):
        tf_fit_num = pd.DataFrame(index=["uang","notelp"],columns=self.kolom)
        
        #uang
        fitur_uang = []
        fitur_notelp = []
        
        for j in self.data_norm:
            pattern_uang = "([0-9]+(k|rb|ribu))|([0-9]+\s+(k|rb|ribu))"
            uang = re.findall(pattern_uang,j,re.IGNORECASE)
            fitur_uang.append(len(uang))

            pattern_notelp = "(\+62|62|0)8[1-9][0-9]"
            notelp = re.findall(pattern_notelp,j)
            fitur_notelp.append(len(notelp))

        tf_fit_num.at["uang"] = fitur_uang
        tf_fit_num.at["notelp"] = fitur_notelp
        # print(tf_fit_text_num)

        term_group = self.kategori.groupby(self.kategori)
        kelas = {}
        for i in np.unique(self.kategori):
            temp = term_group.get_group(i).index.tolist()
            kelas[i] = [j + 1 for j in temp]
        
        total_term = []
        for i in tf_fit_num:
            jumlah = sum(tf_fit_num[i])
            total_term.append(jumlah)

        total_iklan = 0
        index_iklan = kelas["Iklan"]
        for i in index_iklan:
            total_iklan += total_term[i-1]
        # print(total_iklan)

        total_bukan = 0
        index_bukan = kelas["Bukan"]
        for i in index_bukan:
            total_bukan += total_term[i-1]
        # print(total_bukan)

        peluang_term_num = pd.DataFrame(index=["uang","notelp"], columns=["Iklan","Bukan"])
        peluang_term_num["Iklan"] = (tf_fit_num[index_iklan].sum(axis=1) + 1)/(total_iklan + len(tf_fit_num))
        peluang_term_num["Bukan"] = (tf_fit_num[index_bukan].sum(axis=1) + 1)/(total_bukan + len(tf_fit_num))
        
        dict_peluang_term_num = {}
        
        for i in range(len(self.list_kelas)):
            for j in self.list_kelas:
                key = peluang_term_num.index + '|' + j
                val = peluang_term_num[j]
                merg = dict(zip(key, val))
                dict_peluang_term_num.update(merg)

        return dict_peluang_term_num
    
    def peluang_fitur_text(self):
        tf_fit_text = pd.DataFrame(index=["waktu","link"],columns=self.kolom)

        fitur_waktu = []
        fitur_link = []

        for j in self.data_norm:
            pattern_waktu = "(\s|[1-9])(bulan|hari|minggu|tahun)"
            waktu = re.findall(pattern_waktu,j,re.IGNORECASE)
            fitur_waktu.append(len(waktu))

            pattern_link = "http\S"
            link = re.findall(pattern_link,j,re.IGNORECASE)
            fitur_link.append(len(link))

        tf_fit_text.at["waktu"] = fitur_waktu
        tf_fit_text.at["link"] = fitur_link
        # print(tf_fit_text_num)

        term_group = self.kategori.groupby(self.kategori)
        kelas = {}
        for i in np.unique(self.kategori):
            temp = term_group.get_group(i).index.tolist()
            kelas[i] = [j + 1 for j in temp]
        
        total_term = []
        for i in tf_fit_text:
            jumlah = sum(tf_fit_text[i])
            total_term.append(jumlah)

        total_iklan = 0
        index_iklan = kelas["Iklan"]
        for i in index_iklan:
            total_iklan += total_term[i-1]
        # print(total_iklan)

        total_bukan = 0
        index_bukan = kelas["Bukan"]
        for i in index_bukan:
            total_bukan += total_term[i-1]
        # print(total_bukan)

        peluang_term_text = pd.DataFrame(index=["waktu","link"], columns=["Iklan","Bukan"])
        peluang_term_text["Iklan"] = (tf_fit_text[index_iklan].sum(axis=1) + 1)/(total_iklan + len(tf_fit_text))
        peluang_term_text["Bukan"] = (tf_fit_text[index_bukan].sum(axis=1) + 1)/(total_bukan + len(tf_fit_text))
        
        dict_peluang_term_text = {}
        
        for i in range(len(self.list_kelas)):
            for j in self.list_kelas:
                key = peluang_term_text.index + '|' + j
                val = peluang_term_text[j]
                merg = dict(zip(key, val))
                dict_peluang_term_text.update(merg)

        return dict_peluang_term_text