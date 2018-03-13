# Belajar Machine Learning

_Belajar Machine Learning_ adalah salah satu project komunitas _Artificial Intelligence ID_ yang berfokus pada penyediaan sumber belajar Machine Learning.

Tujuan utama _Belajar Machine Learning_ adalah memperbanyak sumber belajar Machine Learning berbahasa Indonesia dan membuat sebuah [Karya Referensi](https://id.wikipedia.org/wiki/Karya_referensi) Machine Learning yang nantinya akan menjadi sumber referensi bagi para pembelajar Machine Learning di Indonesia. Harapan _Belajar Machine Learning_ adalah semakin banyak teman-teman di Indonesia yang berkecimpung dibidang Machine Learning atau mengaplikasikan Machine Learning ke bidang lain. 

Untuk saat ini yang Belajar Machine Learning lakukan adalah menulis ulang artikel tutorial Machine Learning berbahasa Inggris kedalam Bahasa Indonesia dan menulis, merevisi atau menggabungkan potongan potongan artikel yang berisi tentang penjelasan, konsep dan materi Machine Learning kedalam sebuah kategori khusus.

Ini akan menjadi project jangka panjang, anda bisa menjadi bagian dari proses dengan cara ikut berkontribusi pada project ini.

## Bagaimana Saya bisa Berkontribusi dan bagaimana cara kerjanya?

Mudah saja, anda bisa berkontribusi kedalam project sebagai seorang Tutor, Explainer dan Editor.

Seorang Tutor bisa menulis artikel tutorial kedalam Bahasa Indonesia, jika ada penjelasan panjang lebar tentang suatu topik atau konsep yang berkaitan dengan tutorial maka seorang Tutor juga bisa merangkap sebagai seorang Explainer dengan cara memotong bagian penjelasan dan menuliskannya kembali artikel penjelasan tersebut ke kategori khusus atau meminta pada seorang Explainer untuk menuliskan penjelasan tersebut kedalam kategori penjelasan.

Seorang Explainer menulis artikel penjelasan kedalam bahasa indonesia, jika ada penjelasan yang rasanya harus dibuat tutorial (tutorial yang tidak sederhana), maka seorang Explainer juga bisa merangkap sebagai seorang Tutor dengan cara menuliskan artikel tutorial di kategori tutorial atau meminta seorang Tutor untuk menuliskan tutorial tersebut kedalam kategori tutorial.

Terkadang sering terjadi kekeliruan, duplikasi atau kesalahan dalam penulisan atau materi sebuah tutorial dan penjelasan, maka seorang Editor bisa menuliskan, merevisi, atau menggabungkan artikel yang sudah ada menjadi sebuah artikel baru yang lebih lengkap, lebih dapat dipercaya dan lebih terstruktur.

## Apa tujuan dari cara kerja ini?
Tujuannya adalah untuk efisien dan efektif.

dengan menuliskan, merevisi atau menggabungkan potongan potongan artikel yang berisi tentang penjelasan, konsep dan materi, maka secara tidak langsung kita sudah membuat sebuah potongan potongan dokumen yang lebih besar dan terstruktur yang nantinya akan digabungkan lagi hingga akhirnya bisa menjadi sebuah karya referensi.

dengan menuliskan sebuah tutorial, maka orang orang yang bisa belajar berdasarkan melakukan praktek dan langsung memahami topik yang dijelaskan, ini lebih baik daripada anda membaca puluhan halaman buku dan menunggu saat yang tepat untuk mulai bereksperimen dengan Machine Learning.

dan juga, jika seorang pembelajar merasa kebingungan dengan penjelasan konsep yang diajarkan, maka ia bisa dengan mudahnya mencari tutorial terkait yang muncul di pranala dalam penjelasan tersebut, begitu juga sebaliknya.

## Cara Penulisan

Sebelum anda menulis tutorial atau penjelasan konsep, buat issue baru [disini](https://github.com/ai-id/belajar.machinelearning.id/issues/new) dengan judul tutorial atau konsep yang akan anda tulis atau jelaskan. Lalu beri penjelasan singkat beserta link ke tutorial berbahasa Inggris yang akan anda tulis jika ada.

Langkah-langkah untuk menulis tutorial baru:

1. Fork project ini, lalu clone di localmu (jika belum)
    
        # contoh
        git clone https://github.com/pyk/belajar.machinelearning.id.git
        cd belajar.machinelearning.id

2. Install [jekyll](https://jekyllrb.com) (jika belum)

        gem install jekyll bundler

3. Buat file baru di `src/_post/` dengan format `TAHUN-BULAN-TANGGAL-JUDUL`.
4. Pada awal file tambahkan format ini, diikuti materi tutorialnya:

        ---
        layout: post
        title:  "Judul Tutorial/Wiki"
        date:   2017-04-23 01:58:50 +0700
        categories: tutorial/wiki
        ---

5. Jalanin `bundle exec jekyll serve` untuk melihat hasil tulisannya, 
    dan `bundle exec jekyll build` untuk menggenerate HTML nya. 
6. Commmit perubahan di git, lalu kirim pull request.

## Diskusi

Gabung di grub [Artificial Intelligence Indonesia](https://www.facebook.com/groups/381957058844611/) untuk mendiskusikan
project ini.
