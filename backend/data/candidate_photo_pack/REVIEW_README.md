# ShelfLens Candidate Photo Pack

Bu klasor model egitimi icin dogrudan kullanilacak nihai dataset degil; insan kontrolu icin aday foto paketidir.

## Klasorler

- `product_candidates/`: SKU bazli urun/ambalaj referans adaylari.
- `shelf_candidates/`: karisik raf veya market raf goruntuleri.
- `product_candidates_contact_sheet.jpg`: urun adaylarini hizli gozden gecirme onizlemesi.
- `shelf_candidates_contact_sheet.jpg`: raf adaylarini hizli gozden gecirme onizlemesi.
- `manifest.csv` ve `manifest.json`: indirilen ana kaynaklarin URL ve metadata bilgileri.

## Kaynaklar

- Product reference adaylari: Open Food Facts urun gorselleri.
- Shelf adaylari: Wikimedia Commons kategori ve dosya gorselleri.
- Eksik kalan Coca-Cola/Fanta/Sprite/Red Bull referanslari daha once Open Food Facts'ten indirilmis yerel referanslardan bu pakete kopyalandi.

## Kontrol Kriteri

Sil veya kullanma:

- Ambalaj net degilse.
- Urun hedef SKU degilse.
- Raf fotografi icecek rafi degilse.
- Tek urun yerine alakasiz nesne/arka plan agirliktaysa.
- Filigranli, cok karanlik, cok bulaniktir.

Koru:

- Urun etiketi okunuyor.
- Aynı SKU farkli aci/isik/ambalaj formunda gorunuyor.
- Raf fotografi birden fazla icecek markasi/SKU iceriyor.

Kontrolden sonra iyi olanlari asama asama `backend/data/references` ve aktif learning akisine alalim.
