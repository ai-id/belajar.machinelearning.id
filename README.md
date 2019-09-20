Selamat datang di halaman github machine learning id, proyek ini di inisiasi oleh artificial intelligence indonesia (ai-id), tujuan dibuatnya situs ini adalah untuk memperbanyak sumber belajar dan menjadi wadah pembelajaran machine learning di indonesia.

Menulis di markdown cukup rumit bagi yang baru kenal, namun sebenarnya secara teknis lebih efisien menulis markdown ketimbang menulis di platform blog terkenal misalnya seperti wordpress, medium ataupun blogger.

## Cara Kontribusi

Langkah-langkah untuk menulis tutorial baru:

1. Clone project ini di local repo.
    
        # contoh
        git clone https://github.com/ai-id/belajar.machinelearning.id.git
        cd belajar.machinelearning.id

2. Untuk mendaftar sebagai penulis edit file `/_data/authors.yml` dan masukkan data diri mengikuti format berikut, pastikan nama pengguna anda tidak duplikat dengan nama pengguna penulis lain agar nama anda muncul.

        nama_pengguna: 
            name: Nama Anda
            email: email@website.com
            site: https://website.com
            bio: deskripsi singkat.
            image: /data/headshot/nama-anda.jpg

3. lalu upload foto wajah ke `/data/headshot/` dengan format `nama-anda.jpg` atau `nama-anda.png` dan buatlah folder sesuai `nama_pengguna` anda di folder `/images` untuk meletakkan file gambar anda.

2. Buat file baru di `src/_post/` dengan format `TAHUN-BULAN-TANGGAL-JUDUL`.

        #contoh
        2019-09-20-judul-post.markdown

3. Pada awal baris file tambahkan format ini, diikuti materi tutorialnya:
        
        ---
        layout: post
        title:  "Judul Post"
        date:   2017-04-23 01:58:50 +0700
        categories: tutorial/wiki
        author: nama_anda 
        ke `_data/authors.yml`
        ---

4. Commmit perubahan di git, lalu kirim pull request.

## Referensi Menulis Tag Pada Markdown

        https://www.petanikode.com/markdown-pemula/
        https://www.markdownguide.org/cheat-sheet/

## Diskusi

Gabung di grub [Artificial Intelligence Indonesia](https://www.facebook.com/groups/381957058844611/) untuk mendiskusikan
project ini.


__themes jekyll oleh [artemsheludko]( https://github.com/artemsheludko )__
