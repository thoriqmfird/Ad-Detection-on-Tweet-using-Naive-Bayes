import re

class Testing(object):
    def __init__(self,kelas, norm_uji, data_uji, prior, likelihood_bow, likelihood_text, likelihood_num):
        self.kelas = kelas
        self.norm_uji = norm_uji
        self.data_uji = data_uji
        self.prior = prior
        self.likelihood_bow = likelihood_bow
        self.likelihood_text = likelihood_text
        self.likelihood_num = likelihood_num
        self.hasil_bow = self.clasify_bow()
        self.hasil_text = self.clasify_text()
        self.hasil_num = self.clasify_num()


    def clasify_bow(self):
        hasil_kali_likel_bow = [[]for g in range(len(self.kelas))]
        
        for i in range(len(self.kelas)):
            for j in range(len(self.data_uji)):
                perkalian = 1
                for k in self.data_uji[j]:
                    key = k + '|' + self.kelas[i]
                    if key in self.likelihood_bow:
                        perkalian *= self.likelihood_bow[key]
                    else:
                        perkalian *= self.likelihood_bow[("default|"+self.kelas[i])]
                hasil_kali_likel_bow[i].append(perkalian)
        # print(hasil_kali_likel_bow)
        return hasil_kali_likel_bow             #array2d(bukan,iklan)

    def clasify_text(self):
        hasil_kali_likel_text = [[]for g in range(len(self.kelas))]     

        for i in range(len(self.kelas)):
            for j in self.norm_uji:
                hasil_kali = 1
                if(re.search("(\s|[1-9])(bulan|hari|minggu|tahun)", j, re.IGNORECASE)):
                    hasil_kali *= ((self.likelihood_text["waktu|"+self.kelas[i]] ** len(re.findall("(\s|[1-9])(bulan|hari|minggu|tahun)",j,re.IGNORECASE))))
                if(re.search("http\S", j, re.IGNORECASE)):
                    hasil_kali *= ((self.likelihood_text["link|"+self.kelas[i]] ** len(re.findall("http\S",j,re.IGNORECASE))))
                hasil_kali_likel_text[i].append(hasil_kali)
        # print(hasil_kali_likel_text)
        return hasil_kali_likel_text

    def clasify_num(self):
        hasil_kali_likel_num = [[]for g in range(len(self.kelas))]     

        for i in range(len(self.kelas)):
            for j in self.norm_uji:
                hasil_kali = 1
                if(re.search("([0-9]+(k|rb|ribu))|([0-9]+\s+(k|rb|ribu))", j, re.IGNORECASE)):
                    hasil_kali *= ((self.likelihood_num["uang|"+self.kelas[i]] ** len(re.findall("([0-9]+(k|rb|ribu))|([0-9]+\s+(k|rb|ribu))",j,re.IGNORECASE))))
                if(re.search("(\+62|62|0)8[1-9][0-9]{6,9}", j)):
                    hasil_kali *= ((self.likelihood_num["notelp|"+self.kelas[i]] ** len(re.findall("(\+62|62|0)8[1-9][0-9]{6,9}",j))))
                hasil_kali_likel_num[i].append(hasil_kali)
        # print(hasil_kali_likel_num)
        return hasil_kali_likel_num

    def naive_bayes(self):
        posterior = [[]for g in range(len(self.kelas))]
        for i in range(len(self.kelas)):
            for j in range(len(self.hasil_bow[i])):
                hasil_kali = self.prior[self.kelas[i]] * self.hasil_bow[i][j] * self.hasil_text[i][j] * self.hasil_num[i][j] 
                posterior[i].append(hasil_kali)
        # print(posterior)

        hasil_klasifikasi = []
        for i in range(len(posterior[0])):
            hasil_klasifikasi.append("Iklan") if posterior[0][i] < posterior[1][i] else hasil_klasifikasi.append("Bukan")

        return hasil_klasifikasi    

                


    # uang = ((len(re.findall("([0-9]+(k|rb|ribu))|([0-9]+\s+(k|rb|ribu))",j,re.IGNORECASE))) * self.likelihood_text_num["uang|"+self.kelas[i]])
