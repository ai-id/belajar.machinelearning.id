# Belajar Machine Learning

Tempat belajar Machine Learning dari Nol.

Tujuan dibuatnya situs ini adalah untuk memperbanyak sumber belajar Machine
Learning berbahasa Indonesia. Harapan kami adalah semakin banyak teman-teman 
di Indonesia yang paham dan bisa menerapkan Machine Learning. 

Fokus utama kami saat ini adalah menulis kembali tutorial/materi yang
berbahasa Inggris ke bahasa Indonesia dilengkapi dengan penjelasan 
tambahan untuk memperjelas konsep yang sedang dibahas dan untuk mempermudah 
pemahaman.

Project ini masih baru mulai, jadi belum banyak sumber belajar berbahasa
Indonesia yang dipublish.

## Mari Berkontribusi

Semua orang bisa membantu. Sebelum kamu menulis tutorial atau penjelasan
konsep, buat issue baru [disini](https://github.com/ai-id/belajar.machinelearning.id/issues/new) dengan judul tutorial atau konsep yang akan
kamu tulis atau jelaskan. Lalu beri pengem install jekyll bundlerjelasan singkat beserta link ke tutorial
berbahasa Inggris yang akan kamu tulis jika ada.

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
        title:  "Judul Tutorial"
        date:   2017-04-23 01:58:50 +0700
        categories: tutorial
        ---

5. Jalanin `bundle exec jekyll serve` untuk melihat hasil tulisannya, 
    dan `bundle exec jekyll build` untuk menggenerate HTML nya. 
6. Commmit perubahan di git, lalu kirim pull request.

## Diskusi

Gabung di grub [Artificial Intelligence Indonesia](https://www.facebook.com/groups/381957058844611/) untuk mendiskusikan
project ini.
