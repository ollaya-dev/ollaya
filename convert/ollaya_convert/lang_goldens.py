"""Golden fixtures and generated tables for the `ollaya-lang` crate, straight from `laya.lang`.

    cd convert && uv run python -m ollaya_convert.lang_goldens

Writes, under crates/ollaya-lang/:

  tests/fixtures/route.jsonl       one JSON line per state:
                                   {"id", "state",
                                    "analysis": laya.lang.analyse(state),
                                    "latin": laya.lang.latin_profile(state_text(state)),
                                    "route": {"model", "reason"},      # Router()
                                    "route_ml": {"model", "reason"},   # Router(default="multilingual"),
                                                                       # only where it differs
                                    "floats": [repr(x), ...]}          # every float of analysis
                                                                       # and latin, in order
  tests/fixtures/lang_code.jsonl   {"code", "english"}: laya.router._english_from_code
  tests/fixtures/chars.json        Python's character predicates at every table boundary
  src/tables.rs                    the data the port reads from laya and CPython: stopwords,
                                   diacritics, and the character classes behind `str.isalpha` and
                                   `re` for this interpreter's Unicode version

States are round-tripped through JSON before analysis, so each record describes exactly the value
the Rust side parses. `floats` repeats the float fields as Python's repr, which Rust parses exactly
(serde_json's default number parser may be one ulp off), so the test can compare them bit for bit.
"""
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterator, List, Tuple

import laya
from laya import lang
from laya.router import Router, _english_from_code

from . import cases

CRATE = os.path.join(os.path.dirname(__file__), "..", "..", "crates", "ollaya-lang")

# ---------------------------------------------------------------------------------------- corpus

# A customer-support message per language, a few sentences each. `True` marks Latin script, whose
# accent-stripped forms are added as well.
TEXTS: Dict[str, Tuple[bool, List[str]]] = {
    "en": (True, [
        "I was charged twice for my subscription this month.",
        "Could you please refund the duplicate payment?",
        "The invoice number is 4411 and it was paid on March 3rd.",
        "We have been customers for five years and this is the first time it has happened.",
    ]),
    "en_chat": (True, [
        "hey, login's broken again lol",
        "can't reset my password, the link just 404s",
        "pls fix asap, demo at 2pm",
    ]),
    "fr": (True, [
        "Bonjour, j'ai été débité deux fois pour mon abonnement ce mois-ci.",
        "Pouvez-vous rembourser le paiement en double ?",
        "Le numéro de facture est 4411.",
        "Nous sommes clients depuis cinq ans et c'est la première fois que cela arrive.",
    ]),
    "es": (True, [
        "Hola, me cobraron dos veces la suscripción este mes.",
        "¿Pueden devolverme el pago duplicado?",
        "El número de factura es 4411.",
        "Somos clientes desde hace cinco años y es la primera vez que pasa algo así.",
    ]),
    "pt_br": (True, [
        "Olá, fui cobrado duas vezes pela assinatura este mês.",
        "Vocês podem estornar o pagamento duplicado?",
        "O número da nota fiscal é 4411.",
        "Somos clientes há cinco anos e é a primeira vez que isso acontece.",
        "Voce pode me mandar a nota fiscal?",
        "Deu erro 500 no endpoint de login depois do update",
    ]),
    "pt_pt": (True, [
        "Bom dia, fui cobrado em duplicado pela assinatura deste mês.",
        "Podem devolver-me o valor cobrado a mais?",
        "Já enviei três emails e ainda não tive resposta.",
    ]),
    "it": (True, [
        "Buongiorno, mi è stato addebitato due volte l'abbonamento questo mese.",
        "Potete rimborsare il pagamento doppio?",
        "Il numero della fattura è 4411.",
        "Siamo clienti da cinque anni ed è la prima volta che succede.",
    ]),
    "de": (True, [
        "Hallo, mir wurde das Abonnement diesen Monat zweimal berechnet.",
        "Können Sie die doppelte Zahlung bitte erstatten?",
        "Die Rechnungsnummer ist 4411.",
        "Wir sind seit fünf Jahren Kunden und das ist zum ersten Mal passiert.",
    ]),
    "nl": (True, [
        "Hallo, mijn abonnement is deze maand twee keer afgeschreven.",
        "Kunnen jullie de dubbele betaling terugstorten?",
        "Het factuurnummer is 4411.",
        "We zijn al vijf jaar klant en dit is de eerste keer dat het gebeurt.",
    ]),
    "ro": (True, [
        "Bună ziua, am fost taxat de două ori pentru abonament luna aceasta.",
        "Puteți să returnați plata dublă?",
        "Numărul facturii este 4411.",
        "Vreau să anulez abonamentul, dar nu găsesc butonul.",
    ]),
    "pl": (True, [
        "Dzień dobry, w tym miesiącu zostałem obciążony dwa razy za subskrypcję.",
        "Czy możecie zwrócić podwójną płatność?",
        "Numer faktury to 4411.",
        "Jesteśmy klientami od pięciu lat i zdarza się to po raz pierwszy.",
    ]),
    "cs": (True, [
        "Dobrý den, tento měsíc mi bylo předplatné účtováno dvakrát.",
        "Můžete mi prosím vrátit duplicitní platbu?",
        "Jsme zákazníky už pět let a stalo se to poprvé.",
    ]),
    "sk": (True, [
        "Dobrý deň, tento mesiac mi bolo predplatné účtované dvakrát.",
        "Môžete mi prosím vrátiť duplicitnú platbu?",
        "Číslo faktúry je 4411.",
    ]),
    "hu": (True, [
        "Jó napot, ebben a hónapban kétszer vonták le az előfizetés díját.",
        "Vissza tudnák téríteni a dupla terhelést?",
        "Öt éve vagyunk ügyfelek, és ez most fordul elő először.",
    ]),
    "sv": (True, [
        "Hej, jag har debiterats två gånger för mitt abonnemang den här månaden.",
        "Kan ni återbetala den dubbla betalningen?",
        "Vi har varit kunder i fem år och det här är första gången det händer.",
    ]),
    "da": (True, [
        "Hej, jeg er blevet trukket to gange for mit abonnement denne måned.",
        "Kan I refundere den dobbelte betaling?",
        "Vi har været kunder i fem år, og det er første gang, det sker.",
    ]),
    "nb": (True, [
        "Hei, jeg har blitt belastet to ganger for abonnementet denne måneden.",
        "Kan dere refundere den doble betalingen?",
        "Vi har vært kunder i fem år, og dette er første gang det skjer.",
    ]),
    "fi": (True, [
        "Hei, minulta on veloitettu tilauksesta kahdesti tässä kuussa.",
        "Voitteko palauttaa kaksinkertaisen maksun?",
        "Olemme olleet asiakkaita viisi vuotta, ja tämä tapahtuu ensimmäistä kertaa.",
    ]),
    "et": (True, [
        "Tere, mind on selle kuu tellimuse eest kaks korda arveldatud.",
        "Kas saaksite topeltmakse tagastada?",
        "Arve number on 4411.",
    ]),
    "lv": (True, [
        "Labdien, šomēnes par abonementu man tika iekasēta maksa divreiz.",
        "Vai varat atmaksāt dubulto maksājumu?",
        "Rēķina numurs ir 4411.",
    ]),
    "lt": (True, [
        "Laba diena, šį mėnesį už prenumeratą man buvo nuskaityta du kartus.",
        "Ar galite grąžinti dvigubą mokėjimą?",
        "Sąskaitos numeris yra 4411.",
    ]),
    "tr": (True, [
        "Merhaba, bu ay aboneliğim için iki kez ücret alındı.",
        "Mükerrer ödemeyi iade edebilir misiniz?",
        "Fatura numarası 4411.",
        "Beş yıldır müşteriniziz ve bu ilk kez oluyor.",
        "Para iadesi için ne yapmam gerekiyor?",
    ]),
    "vi": (True, [
        "Xin chào, tháng này tôi bị trừ tiền hai lần cho gói đăng ký.",
        "Bạn có thể hoàn lại khoản thanh toán bị trùng không?",
        "Số hóa đơn là 4411.",
        "Chúng tôi đã là khách hàng năm năm và đây là lần đầu tiên chuyện này xảy ra.",
    ]),
    "id": (True, [
        "Halo, bulan ini saya ditagih dua kali untuk langganan saya.",
        "Bisakah Anda mengembalikan pembayaran ganda tersebut?",
        "Nomor tagihannya adalah 4411.",
        "Kami sudah menjadi pelanggan selama lima tahun dan ini pertama kalinya terjadi.",
    ]),
    "ms": (True, [
        "Helo, saya telah dicaj dua kali untuk langganan bulan ini.",
        "Bolehkah anda memulangkan bayaran berganda itu?",
    ]),
    "tl": (True, [
        "Kumusta, dalawang beses akong siningil para sa subscription ngayong buwan.",
        "Maaari ba ninyong ibalik ang dobleng bayad?",
    ]),
    "sw": (True, [
        "Habari, nimetozwa mara mbili kwa usajili wangu mwezi huu.",
        "Je, mnaweza kunirudishia malipo yaliyorudiwa?",
        "Nambari ya ankara ni 4411.",
    ]),
    "hr": (True, [
        "Dobar dan, ovaj mjesec pretplata mi je naplaćena dvaput.",
        "Možete li mi vratiti dvostruku uplatu?",
        "Vaši smo korisnici već pet godina i ovo se događa prvi put.",
    ]),
    "sl": (True, [
        "Pozdravljeni, ta mesec mi je bila naročnina zaračunana dvakrat.",
        "Ali mi lahko vrnete podvojeno plačilo?",
    ]),
    "ca": (True, [
        "Hola, aquest mes m'han cobrat dues vegades la subscripció.",
        "Em podeu retornar el pagament duplicat?",
        "El número de factura és 4411.",
    ]),
    "af": (True, [
        "Hallo, ek is hierdie maand twee keer vir my intekening gehef.",
        "Kan julle asseblief die dubbele betaling terugbetaal?",
    ]),
    "ga": (True, [
        "Dia duit, gearradh táille orm faoi dhó as mo shíntiús an mhí seo.",
        "An féidir libh an íocaíocht dhúbailte a aisíoc?",
    ]),
    "eu": (True, [
        "Kaixo, hilabete honetan bi aldiz kobratu didate harpidetza.",
        "Itzul al diezadakezue ordainketa bikoitza?",
    ]),
    "hi": (False, [
        "नमस्ते, इस महीने मेरी सदस्यता के लिए मुझसे दो बार शुल्क लिया गया।",
        "क्या आप दोहरा भुगतान वापस कर सकते हैं?",
        "चालान संख्या 4411 है।",
        "हम पाँच साल से ग्राहक हैं और ऐसा पहली बार हुआ है।",
    ]),
    "mr": (False, ["नमस्कार, या महिन्यात माझ्या सदस्यत्वासाठी दोनदा शुल्क आकारले गेले."]),
    "bn": (False, [
        "নমস্কার, এই মাসে আমার সাবস্ক্রিপশনের জন্য দুবার টাকা কাটা হয়েছে।",
        "আপনি কি দ্বিগুণ অর্থ ফেরত দিতে পারবেন?",
    ]),
    "pa": (False, ["ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਇਸ ਮਹੀਨੇ ਮੇਰੇ ਤੋਂ ਦੋ ਵਾਰ ਪੈਸੇ ਕੱਟੇ ਗਏ। ਕਿਰਪਾ ਕਰਕੇ ਵਾਪਸ ਕਰੋ।"]),
    "gu": (False, ["નમસ્તે, આ મહિને મારી પાસેથી બે વાર ચાર્જ લેવામાં આવ્યો. કૃપા કરીને રિફંડ આપો."]),
    "or": (False, ["ନମସ୍କାର, ଏହି ମାସରେ ମୋଠାରୁ ଦୁଇଥର ଟଙ୍କା କଟାଯାଇଛି।"]),
    "ta": (False, [
        "வணக்கம், இந்த மாதம் எனது சந்தாவுக்கு இரண்டு முறை கட்டணம் வசூலிக்கப்பட்டது.",
        "இரட்டிப்பு கட்டணத்தைத் திருப்பித் தர முடியுமா?",
    ]),
    "te": (False, ["నమస్కారం, ఈ నెల నా సభ్యత్వానికి రెండుసార్లు డబ్బు తీసుకున్నారు. దయచేసి తిరిగి ఇవ్వండి."]),
    "kn": (False, ["ನಮಸ್ಕಾರ, ಈ ತಿಂಗಳು ನನ್ನ ಚಂದಾದಾರಿಕೆಗೆ ಎರಡು ಬಾರಿ ಶುಲ್ಕ ವಿಧಿಸಲಾಗಿದೆ."]),
    "ml": (False, ["നമസ്കാരം, ഈ മാസം എന്റെ സബ്സ്ക്രിപ്ഷന് രണ്ടുതവണ പണം ഈടാക്കി."]),
    "si": (False, ["ආයුබෝවන්, මෙම මාසයේ මගේ දායකත්වය සඳහා දෙවරක් ගාස්තු අය කර ඇත."]),
    "th": (False, [
        "สวัสดีครับ เดือนนี้ผมถูกเรียกเก็บเงินค่าสมาชิกสองครั้ง",
        "ช่วยคืนเงินส่วนที่ซ้ำได้ไหมครับ หมายเลขใบแจ้งหนี้คือ 4411",
    ]),
    "lo": (False, ["ສະບາຍດີ, ເດືອນນີ້ຂ້ອຍຖືກເກັບເງິນສອງຄັ້ງ."]),
    "km": (False, [
        "សួស្តី ខែនេះខ្ញុំត្រូវបានគិតប្រាក់ពីរដងសម្រាប់ការជាវរបស់ខ្ញុំ។",
        "តើអ្នកអាចបង្វិលប្រាក់វិញបានទេ?",
    ]),
    "my": (False, ["မင်္ဂလာပါ၊ ဒီလမှာ ကျွန်တော့်ဆီက ငွေနှစ်ကြိမ် ဖြတ်တောက်ခံရပါတယ်။"]),
    "bo": (False, ["བཀྲ་ཤིས་བདེ་ལེགས། ཟླ་བ་འདིར་ང་ལ་ཐེངས་གཉིས་རིན་བསྡུས་སོང་།"]),
    "ka": (False, ["გამარჯობა, ამ თვეში გამოწერისთვის ორჯერ ჩამომეჭრა თანხა. შეგიძლიათ დამიბრუნოთ?"]),
    "hy": (False, ["Բարև, այս ամիս բաժանորդագրության համար ինձանից երկու անգամ գումար է գանձվել:"]),
    "am": (False, ["ሰላም፣ በዚህ ወር ለደንበኝነት ምዝገባዬ ሁለት ጊዜ ክፍያ ተወስዶብኛል።"]),
    "el": (False, [
        "Γεια σας, αυτόν τον μήνα χρεώθηκα δύο φορές για τη συνδρομή μου.",
        "Μπορείτε να επιστρέψετε τη διπλή πληρωμή;",
        "Ο αριθμός τιμολογίου είναι 4411.",
    ]),
    "ru": (False, [
        "Здравствуйте, в этом месяце с меня дважды списали плату за подписку.",
        "Можете вернуть двойной платёж?",
        "Номер счёта 4411.",
        "Мы клиенты уже пять лет, и такое случилось впервые.",
    ]),
    "uk": (False, [
        "Добрий день, цього місяця з мене двічі списали оплату за підписку.",
        "Чи можете ви повернути подвійний платіж?",
    ]),
    "bg": (False, ["Здравейте, този месец ми таксуваха абонамента два пъти. Можете ли да възстановите двойното плащане?"]),
    "sr_cyrl": (False, ["Добар дан, овог месеца ми је претплата наплаћена двапут. Можете ли да ми вратите дуплу уплату?"]),
    "kk": (False, ["Сәлеметсіз бе, осы айда жазылым үшін менен екі рет ақша алынды."]),
    "mn": (False, ["Сайн байна уу, энэ сард миний захиалгын төлбөрийг хоёр удаа авсан."]),
    "he": (False, [
        "שלום, החודש חויבתי פעמיים על המנוי שלי.",
        "תוכלו להחזיר את התשלום הכפול? מספר החשבונית הוא 4411.",
    ]),
    "ar": (False, [
        "مرحباً، تم خصم رسوم الاشتراك مني مرتين هذا الشهر.",
        "هل يمكنكم استرداد المبلغ المكرر؟ رقم الفاتورة هو 4411.",
    ]),
    "fa": (False, ["سلام، این ماه دو بار هزینه اشتراک از من کسر شد. آیا می‌توانید پرداخت تکراری را برگردانید؟"]),
    "ur": (False, ["السلام علیکم، اس مہینے میری سبسکرپشن کے لیے مجھ سے دو بار رقم کاٹی گئی۔"]),
    "zh_hans": (False, [
        "你好，这个月我的订阅被扣了两次费。",
        "能否退还重复扣除的款项？发票号码是 4411。",
        "我们已经是五年的老客户了，这是第一次发生这种情况。",
    ]),
    "zh_hant": (False, ["您好，這個月我的訂閱被扣了兩次款。可以退還重複的款項嗎？發票號碼是 4411。"]),
    "ja": (False, [
        "こんにちは。今月、サブスクリプションの料金が二重に請求されました。",
        "重複分を返金していただけますか？請求書番号は4411です。",
        "ありがとうございます",
        "コンピューター サーバー エラー",
    ]),
    "ko": (False, [
        "안녕하세요, 이번 달 구독료가 두 번 청구되었습니다.",
        "중복 결제된 금액을 환불해 주실 수 있나요? 청구서 번호는 4411입니다.",
    ]),
    # Scripts no range claims: they count as "other".
    "mn_mong": (False, ["ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ ᠦᠰᠦᠭ"]),
    "dv": (False, ["ދިވެހި ބަސް އަކީ ރާއްޖޭގެ ބަހެވެ"]),
    "syr": (False, ["ܠܫܢܐ ܣܘܪܝܝܐ ܥܬܝܩܐ"]),
    "chr": (False, ["ᏣᎳᎩ ᎦᏬᏂᎯᏍᏗ ᎠᏍᎦᏯ"]),
    "iu": (False, ["ᐃᓄᒃᑎᑐᑦ ᐅᖃᐅᓯᖅ"]),
    "zgh": (False, ["ⵜⴰⵎⴰⵣⵉⵖⵜ ⵜⴰⵏⴰⵡⴰⵢⵜ"]),
    "nqo": (False, ["ߒߞߏ ߞߊ߲"]),
}

EXTRA_TEXTS = {
    # regressions the laya sources call out
    "laya/pt_quero": "Quero cancelar",
    "laya/pt_senha": "Esqueci minha senha",
    "laya/it_fattura": "la fattura",
    "laya/it_fattura_long": "Non ho ricevuto la fattura del mese scorso",
    "laya/es_de_facto": "This is de facto the default en route to Rio de Janeiro",
    "laya/en_us": "Set the locale to en-US and the fallback to en-GB",
    "laya/ro_la_e": "Mă duc la magazin și e foarte aproape de casă",
    "laya/tr_para": "Para iadesi için ne yapmalıyım, çok acil",
    "laya/links": "See github.com and docs.python.org or email user@acme.com and admin@le.la.com",
    "laya/links_prose": "See github.com/org/repo, v1.2.3 and U.S.A. for the release notes",
    "laya/final_period": "Il pacco non è ancora arrivato.",
    "laya/de_masking": "Mein Konto wurde zweimal belastet",
    "laya/en_charged": "I was charged twice",
    # code, identifiers, machine text
    "code/python": "def refund(order_id: int) -> bool:\n    return api.post(f'/orders/{order_id}/refund').ok",
    "code/js": "const total = items.reduce((sum, i) => sum + i.price, 0); console.log(`total: ${total}`);",
    "code/rust": "fn main() { let v: Vec<u32> = (0..10).collect(); println!(\"{:?}\", v); }",
    "code/sql": "SELECT id, email FROM users WHERE created_at > '2024-01-01' AND status = 'active';",
    "code/shell": "curl -sS https://api.example.com/v1/items | jq '.data[] | select(.la == \"que\")'",
    "code/yaml": "server:\n  host: 0.0.0.0\n  port: 8080\n  tls: { enabled: true }",
    "code/html": "<div class=\"alert\">Le paiement a échoué</div><a href=\"https://pay.example.fr\">Réessayer</a>",
    "code/stack": "Traceback (most recent call last):\n  File \"app.py\", line 12, in <module>\nKeyError: 'user_id'",
    "code/java": "java.lang.NullPointerException at com.acme.billing.InvoiceService.charge(InvoiceService.java:88)",
    "code/log": "2024-05-01T12:00:03Z ERROR [billing] charge failed: card_declined (code=51) req=ab12-cd34",
    "code/json_text": "{\"message\": \"Bonjour, je veux annuler mon abonnement\", \"lang\": \"fr\"}",
    "code/diff": "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1,3 +1,4 @@\n-use std::fmt;\n+use std::fmt::{self, Write};",
    "code/regex": r"^[\w.+-]+@[\w-]+\.[\w.-]+$",
    "code/ident_edge": "a..b a.b.c -x.y- @que .la para.es le@la e.g. i.e. x-y.z_w foo_bar.baz",
    "url/one": "https://github.com/cobanov/ollaya",
    "url/many": "https://example.com/a http://le.la.de/que/es/para www.example.org/path?q=1&lang=fr",
    "url/emails": "user@acme.com support@example.org la@que.es",
    "url/bare_path": "/usr/local/bin/python3 /etc/nginx/sites-enabled/default",
    "num/int": "1234567890",
    "num/floats": "3.14159 2.71828 -1.5e10 0x1F",
    "num/phone": "+1 (555) 123-4567 ext. 89",
    "num/arabic_indic": "٠١٢٣٤٥٦٧٨٩",
    "num/devanagari": "१२३४५",
    "num/fullwidth": "１２３４５",
    "num/roman": "Ⅻ Ⅳ ⅲ",
    "num/super": "² ³ ½ ¾ ①②",
    "num/cjk": "一二三 四五",
    "empty/empty": "",
    "empty/space": " ",
    "empty/whitespace": " \t\n\r\x0b\x0c ",
    "empty/punct": "!!! ??? ... --- ***",
    "empty/symbols": "€ $ £ ¥ © ® ™ ° ± × ÷",
    "emoji/only": "😀😃😄😁",
    "emoji/zwj": "👨‍👩‍👧 👍🏽 🏳️‍🌈",
    "emoji/flags": "🇹🇷🇩🇪🇫🇷",
    "emoji/text": "Worst purchase ever 😡😡 refund pls",
    "emoji/text_fr": "Service génial 😍 merci beaucoup pour tout 🙏",
    "emoji/family": "👨‍👩‍👧 family plan please 🙏🏽",
    "mixed/en_zh_tie": "ab 中文",
    "mixed/zh_en_tie": "中文 ab",
    "mixed/en_el_tie": "ab αβ",
    "mixed/el_other_tie": "αβ ᏣᎳ",
    "mixed/other_el_tie": "ᏣᎳ αβ",
    "mixed/en_mostly": "Please refund the order 注文 for my account",
    "mixed/zh_mostly": "我的 order 被扣了两次费用，请退款",
    "mixed/ru_en": "Привет, the invoice is wrong, пожалуйста исправьте",
    "mixed/ar_en": "مرحبا hello مرحبا",
    "mixed/code_switch": "Ich habe the invoice nicht bekommen, aber the payment ist durch",
    "mixed/en_fr": "I was charged twice. Je veux être remboursé pour la facture de mars.",
    "mixed/fr_en": "Je veux être remboursé. The invoice is wrong and I want a refund please.",
    "mixed/es_pt": "Quiero cancelar la suscripción, não quero mais pagar por isso",
    "uni/turkish_upper": "İSTANBUL ŞUBESİ İÇİN İADE İSTİYORUM, İKİ KEZ ÜCRET ALINDI",
    "uni/dotted_i": "İİİİ the and is are",
    "uni/greek_sigma": "ΟΔΥΣΣΕΥΣ ΚΑΙ ΣΙΣΥΦΟΣ",
    "uni/kelvin_angstrom": "Temperature 300 K and length 5 Å in the lab",
    "uni/sharp_s": "STRASSE ẞ GROẞE Straße",
    "uni/ligatures": "ﬁnance ﬂow ofﬁce",
    "uni/fullwidth": "ｈｅｌｌｏ ｗｏｒｌｄ ｔｈｅ ａｎｄ ｉｓ ＴＨＥ",
    "uni/math_bold": "𝐓𝐡𝐞 𝐪𝐮𝐢𝐜𝐤 𝐛𝐫𝐨𝐰𝐧 𝐟𝐨𝐱",
    "uni/circled": "ⓣⓗⓔ ⓠⓤⓘⓒⓚ ⓑⓡⓞⓦⓝ",
    "uni/superscript_words": "the² and³ is½ are x²y the the",
    "uni/roman_words": "Chapter Ⅻ the and is are",
    "uni/nfd_fr": unicodedata.normalize("NFD", "Café très crème brûlée pour les amis et la famille"),
    "uni/nfd_vi": unicodedata.normalize("NFD", "Tôi muốn hoàn tiền cho đơn hàng này"),
    "uni/zwnj": "می‌توانید",
    "uni/rtl_marks": "‏שלום‎ hello there",
    "uni/bom": "﻿Hello there, the invoice is attached",
    "uni/private_use": "",
    "uni/cjk_ext_b": "𠀀𠀁𠀂 𪜀",
    "uni/cjk_ext_i": "\U0002ebf0\U0002ebf1\U0002ebf2",
    "uni/cjk_ext_i_en": "the invoice \U0002ebf0\U0002ebf1\U0002ebf2\U0002ebf3\U0002ebf4",
    "uni/unicode16_garay": "\U00010d50\U00010d51\U00010d52\U00010d70",
    "uni/unicode16_latin": "ꟋꟌꟚꟜ and the",
    "uni/ipa": "ðə kwɪk braʊn fɒks ɐɑɒ",
    "uni/modifier": "ʰʲʷ ˡ ˢ",
    "uni/titlecase": "ǅ ǈ ǋ ǲ",
    "uni/latin_ext_c": "ⱥⱦ ꜳꜵ",
    "uni/letterlike": "ℌℑℜ ℭ",
    "uni/halfwidth_kana": "ｺﾝﾋﾟｭｰﾀｰ",
    "uni/halfwidth_hangul": "ﾡﾢﾣ",
    "uni/bopomofo": "ㄅㄆㄇㄈ",
    "uni/kana_ext": "ㇰㇱㇲ",
    "uni/cjk_compat": "豈更車",
    "uni/cjk_ext_a": "㐀㐁㐂",
    "uni/hangul_jamo": "ᄀᄁᄂ ꥠꥡ",
    "uni/arabic_forms": "ﭐﭑﭒ ﹰﹱﹲ",
    "uni/arabic_ext_b": "ࡰࡱࡲ",
    "uni/cyrillic_ext": "ꙀꙁꙂ ᲀᲁᲂ",
    "uni/georgian_mtavruli": "ᲐᲑᲒ ⴀⴁⴂ",
    "uni/ethiopic_supp": "ᎀᎁᎂ",
    "uni/greek_ext": "ἀἁἂ ᾀᾁ",
    "uni/latin_ext_add": "ḀḁḂ ỳỹ",
    "uni/combining_only": "́̀̂",
    "uni/nbsp": "the invoice is　late and",
    "uni/control": "the\x00invoice\x1cis\x7flate",
    "uni/feminine_ordinal": "1ª 2º µ",
}


def strip_accents(s: str) -> str:
    """What a mail client or an ASCII-normalising pipeline leaves of accented text."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if unicodedata.category(c) != "Mn")


def language_states() -> Iterator[Tuple[str, Any]]:
    for code, (latin, sents) in TEXTS.items():
        full = " ".join(sents)
        for i, s in enumerate(sents):
            yield "lang/%s/s%d" % (code, i), s
        words = sents[0].split()
        for k in range(1, 6):
            yield "lang/%s/w%d" % (code, k), " ".join(words[:k])
        yield "lang/%s/full" % code, full
        yield "lang/%s/upper" % code, full.upper()
        yield "lang/%s/dict" % code, {"subject": sents[0], "body": " ".join(sents[1:]), "priority": 2}
        yield "lang/%s/conv" % code, [{"role": "user" if i % 2 == 0 else "agent", "content": s}
                                      for i, s in enumerate(sents)]
        if code in ("en", "fr", "tr", "zh_hans", "ar"):
            # past the 4000-character cut
            yield "lang/%s/long" % code, " ".join([full] * (4200 // len(full) + 1))
        if latin:
            stripped = [strip_accents(s) for s in sents]
            for i, s in enumerate(stripped):
                if s != sents[i]:
                    yield "lang/%s/stripped/s%d" % (code, i), s
            yield "lang/%s/stripped/full" % code, " ".join(stripped)
            yield "lang/%s/stripped/dict" % code, {"message": " ".join(stripped)}


def case_states() -> Iterator[Tuple[str, Any]]:
    for name, state in cases.STATES.items():
        yield "cases/%s" % name, state
    yield "cases/long_state", cases.LONG_STATE
    yield "cases/empty_state", ""
    yield "cases/edge_questions", cases.EDGE_QUESTIONS
    for pname, qs in cases.preset_sets().items():
        yield "cases/preset/%s" % pname, qs
        for qid, q in qs.items():
            yield "cases/preset/%s/%s" % (pname, qid), q["instructions"]
    for cid, state, _ in cases.typed_decisions():
        yield cid, state


def script_boundary_states() -> Iterator[Tuple[str, Any]]:
    """Every letter just inside and just outside each script range laya lists."""
    bounds = [("latin", ((0x0000, 0x024F), (0x1E00, 0x1EFF), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A)))]
    bounds += lang._SCRIPT_RANGES
    for name, ranges in bounds:
        cps = {cp for lo, hi in ranges for cp in (lo - 1, lo, lo + 1, (lo + hi) // 2, hi - 1, hi, hi + 1)}
        for cp in sorted(cps):
            if 0 <= cp < 0x110000 and chr(cp).isalpha():
                yield "script/%s/%04X" % (name, cp), chr(cp) * 3
    # one state per range holding its letters (the first 3000 of them)
    for name, ranges in lang._SCRIPT_RANGES:
        letters = "".join(chr(cp) for lo, hi in ranges for cp in range(lo, hi + 1) if chr(cp).isalpha())
        yield "script/%s/all" % name, letters[:3000]


def rate_states() -> Iterator[Tuple[str, Any]]:
    """Diacritic rates and script fractions around the thresholds and the %.0f rounding ties."""
    for n in range(30, 71, 2):
        base = ("We met at the cafe for a quick coffee and then went home to rest " * 2)[:n - 1]
        yield "rate/threshold/%d" % n, "é" + base
    for d, n in ((1, 8), (3, 8), (5, 8), (1, 16), (1, 32), (1, 40), (7, 200), (1, 50), (1, 51),
                 (2, 99), (5, 99), (3, 7), (1, 3), (9, 200), (11, 400), (1, 400), (1, 401)):
        yield "rate/diac/%d_%d" % (d, n), "é" * d + "x" * (n - d)
        yield "rate/diac_words/%d_%d" % (d, n), " ".join(["é" * d, "x" * max(1, n - d - 4), "y", "z", "w"])
    for k in range(1, 13):
        for m in range(0, k + 1):
            yield "rate/greek/%d_%d" % (k, m), "α" * k + " " + "a" * m
            if m:
                yield "rate/greek_rev/%d_%d" % (k, m), "a" * m + " " + "α" * k
    for k, m in ((31, 1), (3, 29), (5, 3), (7, 1), (13, 3), (1, 7), (5, 11)):
        yield "rate/other/%d_%d" % (k, m), "Ꮳ" * k + " " + "b" * m
    # Odd multiples of 1/32 are the only exact ties of round(x, 4), which Python breaks to even.
    for k in range(1, 32, 2):
        yield "rate/tie32/%d" % k, "α" * k + " " + "a" * (32 - k)
        yield "rate/diac_tie32/%d" % k, "é" * k + "x" * (32 - k)


def nested_states() -> Iterator[Tuple[str, Any]]:
    fr = " ".join(TEXTS["fr"][1])
    for depth in range(0, 10):
        d: Any = fr
        for _ in range(depth):
            d = {"n": d}
        yield "nested/dict_depth/%d" % depth, d
        l: Any = fr
        for _ in range(depth):
            l = [l]
        yield "nested/list_depth/%d" % depth, l
    # a different language at every level: which ones survive the depth cut decides the route
    langs = ["en", "fr", "de", "es", "pt_br", "it", "nl", "ro", "tr"]
    for start in range(len(langs)):
        order = langs[start:] + langs[:start]
        d = None
        for code in reversed(order):
            d = {"text": " ".join(TEXTS[code][1]), "child": d}
        yield "nested/levels/%s" % order[0], d
    yield "nested/non_string", {"a": 1, "b": True, "c": None, "d": 2.5, "e": [1, 2, [3, False]]}
    yield "nested/keys_only", {"le la que para con": 1, "der die das und": {"ist": 2}}
    yield "nested/empty_dict", {}
    yield "nested/empty_list", []
    yield "nested/empty_strings", ["", "", ""]
    yield "nested/space_strings", [" ", "\n"]
    yield "nested/null", None
    yield "nested/true", True
    yield "nested/false", False
    yield "nested/int", 42
    yield "nested/float", 3.5
    yield "nested/list_of_scalars", [1, 2.5, None, True, "Bonjour tout le monde, merci pour votre aide"]
    yield "nested/ticket", {
        "ticket": {"id": 4411, "subject": TEXTS["es"][1][0], "tags": ["billing", "urgent"],
                   "meta": {"lang": "es", "channel": "email", "vip": False}},
        "messages": [{"from": "customer", "text": s} for s in TEXTS["es"][1][1:]],
    }
    yield "nested/mixed_langs", {"en": TEXTS["en"][1][0], "fr": TEXTS["fr"][1][0], "de": TEXTS["de"][1][0],
                                 "tr": TEXTS["tr"][1][0], "zh": TEXTS["zh_hans"][1][0]}
    yield "nested/english_keys_turkish_values", {"customer_message": TEXTS["tr"][1][0],
                                                 "agent_reply": TEXTS["tr"][1][1]}
    # the 4000-character cut, landing inside a word, on a separator and inside an identifier
    for pad in (3990, 3995, 3997, 3998, 3999, 4000, 4001):
        yield "nested/cut/%d" % pad, {"a": "x" * pad, "b": "le la les des une est pour dans que qui"}
    for pad in (3985, 3990, 3993):
        yield "nested/cut_ident/%d" % pad, "a " * (pad // 2) + "see github.com/le/la for details"
    yield "nested/cut_multi", ["x" * 1999, "y" * 2000, "Je veux être remboursé pour la facture"]
    en = " ".join(TEXTS["en"][1])
    yield "nested/en_then_fr", {"a": (en + " ") * 60, "b": fr}
    yield "nested/fr_then_en", {"a": fr, "b": (en + " ") * 60}
    yield "nested/many_leaves", [{"k": w} for w in ("le la les des une est pour dans que qui " * 40).split()]
    yield "nested/json_string", json.dumps({"message": fr, "lang": "fr"}, ensure_ascii=False)


def salad_states(n: int = 300, seed: int = 20240501) -> Iterator[Tuple[str, Any]]:
    """Random mixes of stopwords, content words, identifiers and symbols: margins and ties."""
    rng = random.Random(seed)
    stop = {lg: sorted(ws) for lg, ws in lang._STOP.items()}
    shared = sorted(lang._SHARED_WORDS)
    pools = [
        (6, [w for lg in stop for w in stop[lg]]),
        (2, shared),
        (2, stop["en"]),
        (2, ["invoice", "refund", "account", "server", "error", "login", "customer", "payment",
             "order", "update", "deploy", "ticket", "factura", "fattura", "Rechnung", "fatura",
             "abonnement", "Konto", "cuenta", "conta", "senha", "cancelar", "hesap", "ödeme"]),
        (1, ["été", "très", "más", "não", "già", "für", "până", "łódź", "çok", "ığdır", "ÉCOLE",
             "žluťoučký", "Ärger", "ÇA", "Ñandú", "şi", "ţară", "æble", "œuvre", "đi", "ő"]),
        (1, ["github.com", "user@acme.com", "v1.2.3", "U.S.A.", "e.g.", "la.com", "@que", "para.es",
             "file_name.py", "x-y.z", "le-la", "que_es", "de.en", "a.", ".la"]),
        (1, ["42", "3.14", "#4411", "—", "!", "?", "(", ")", "½", "x²", "Ⅻ", "\n", ",", "...", "😀"]),
        (1, ["Привет", "γεια", "你好", "שלום", "مرحبا", "नमस्ते", "ᏣᎳᎩ"]),
    ]
    weights = [w for w, _ in pools]
    for i in range(n):
        k = rng.choice([0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 20, 30])
        toks = [rng.choice(rng.choices(pools, weights)[0][1]) for _ in range(k)]
        toks = [t.upper() if rng.random() < 0.08 else t.capitalize() if rng.random() < 0.1 else t
                for t in toks]
        text = ""
        for t in toks:
            text += t + rng.choice([" ", " ", " ", ", ", ". ", "\n", "-", ""])
        yield "salad/%03d" % i, text.strip() if rng.random() < 0.7 else text


def all_states() -> Iterator[Tuple[str, Any]]:
    yield from case_states()
    yield from language_states()
    for name, text in EXTRA_TEXTS.items():
        yield "text/%s" % name, text
    yield from script_boundary_states()
    yield from rate_states()
    yield from nested_states()
    yield from salad_states()


LANG_CODES = [
    "en", "EN", "en-US", "en_US", "en_US.UTF-8", " en ", "\ten\n", "eng", "English", "ENGLISH",
    "english-uk", "english_US", "eng-GB", "en-", "-en", "_en", ".en", ".", "", "   ", "\n", "_", "-",
    "._", "..en", "fr", "fr-CA", "de_DE.UTF-8", "pt-BR", "zh-Hans-CN", "C", "C.UTF-8", "POSIX", "e",
    "enn", "en.", "EN_us", "en@euro", "En_Us.utf8", "en_US.ISO-8859-1", "es-en", "x-en", "e-n",
    "\x1cen\x1f", "　en　", " en ", "​en", "ｅｎ", "İn", "ENİ",
    "\u0085en ", "und", "mul", "tr", "hi-IN", "english.", "en-GB-oxendict",
]

# ---------------------------------------------------------------------------------------- tables


def char_class(ch: str, ident_char: "re.Pattern[str]") -> int:
    """0 none, 1 alpha, 2 numeric (not decimal), 3 decimal: all three predicates follow from it."""
    alpha = ch.isalpha()
    letter = lang._WORD.fullmatch(ch) is not None       # [^\W\d_]
    word = ident_char.fullmatch(ch) is not None and ch != "-"   # \w
    cls = 1 if alpha else 2 if letter else 3 if word and ch != "_" else 0
    # the classes must reproduce each predicate exactly, `_` aside
    assert letter == (cls in (1, 2)), hex(ord(ch))
    assert word == (cls != 0 or ch == "_"), hex(ord(ch))
    assert cls != 3 or ch.isdecimal(), hex(ord(ch))
    return cls


def class_ranges() -> List[Tuple[int, int, int]]:
    ident_char = re.compile(r"[\w-]")
    ranges: List[List[int]] = []
    for cp in range(0x110000):
        cls = char_class(chr(cp), ident_char)
        if not cls:
            continue
        if ranges and ranges[-1][2] == cls and ranges[-1][1] == cp - 1:
            ranges[-1][1] = cp
        else:
            ranges.append([cp, cp, cls])
    return [tuple(r) for r in ranges]


def rust_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def rust_char(c: str) -> str:
    return "'%s'" % c


def write_tables(path: str, ranges: List[Tuple[int, int, int]]) -> None:
    languages = list(lang._STOP)
    consts = [lg.upper() for lg in languages]
    words = sorted({w for ws in lang._STOP.values() for w in ws}, key=lambda w: w.encode())
    for w in words:
        assert (w in lang._SHARED_WORDS) == (sum(w in ws for ws in lang._STOP.values()) > 1)

    def mask(w: str) -> str:
        return " | ".join(c for lg, c in zip(languages, consts) if w in lang._STOP[lg])

    def rows(items: List[str], width: int = 100) -> List[str]:
        lines, line = [], "   "
        for it in items:
            if len(line) + 1 + len(it) + 1 > width:
                lines.append(line)
                line = "   "
            line += " " + it + ","
        if line.strip():
            lines.append(line)
        return lines

    names = {1: "A", 2: "N", 3: "D"}
    out = [
        "//! Data read from laya and CPython. Generated by `python -m ollaya_convert.lang_goldens`; do not",
        "//! edit.",
        "//!",
        "//! laya %s on CPython %s, Unicode %s. `CHAR_CLASSES` is what that interpreter's"
        % (laya.__version__, sys.version.split()[0], unicodedata.unidata_version),
        "//! `str.isalpha` and `re` see, so detection classifies every code point as laya does whatever",
        "//! Unicode version the Rust standard library ships.",
        "",
        "use crate::pystr::CharClass::{self, Alpha as A, Decimal as D, Numeric as N};",
        "",
        "/// Languages with a stopword list, in laya's `_STOP` order: ties go to the earlier one.",
        "pub(crate) const LANGUAGES: [&str; %d] = [%s];" % (len(languages), ", ".join(map(rust_str, languages))),
        "",
    ]
    out += ["const %s: u8 = 1 << %d;" % (c, i) for i, c in enumerate(consts)]
    out += [
        "",
        "/// Every stopword with the languages that list it (one bit per `LANGUAGES` entry), sorted.",
        "#[rustfmt::skip]",
        "pub(crate) static STOPWORDS: &[(&str, u8)] = &[",
    ]
    out += rows(["(%s, %s)" % (rust_str(w), mask(w)) for w in words])
    out += [
        "];",
        "",
        "/// `_NON_EN_DIACRITICS`: letters ordinary English does not use, sorted.",
        "#[rustfmt::skip]",
        "pub(crate) static DIACRITICS: &[char] = &[",
    ]
    out += rows([rust_char(c) for c in sorted(lang._NON_EN_DIACRITICS)])
    out += [
        "];",
        "",
        "/// Maximal code point ranges of each class, sorted. Code points outside them have none.",
        "#[rustfmt::skip]",
        "pub(crate) static CHAR_CLASSES: &[(u32, u32, CharClass)] = &[",
    ]
    out += rows(["(0x%04X, 0x%04X, %s)" % (lo, hi, names[c]) for lo, hi, c in ranges])
    out += ["];", ""]
    with open(path, "w") as f:
        f.write("\n".join(out))


def chars_fixture(ranges: List[Tuple[int, int, int]]) -> Dict[str, List[int]]:
    """Python's predicates on both sides of every range boundary, plus every `str.isspace` char."""
    ident_char = re.compile(r"[\w-]")
    probes = set(range(0x80))
    for lo, hi, _ in ranges:
        probes.update((lo - 1, lo, hi, hi + 1))
    probes = sorted(cp for cp in probes if 0 <= cp < 0x110000 and not 0xD800 <= cp <= 0xDFFF)
    return {
        "probes": probes,
        "alpha": [cp for cp in probes if chr(cp).isalpha()],
        "word_letter": [cp for cp in probes if lang._WORD.fullmatch(chr(cp))],
        "word": [cp for cp in probes if ident_char.fullmatch(chr(cp)) and chr(cp) != "-"],
        "space": [cp for cp in range(0x110000) if chr(cp).isspace()],
    }


# ---------------------------------------------------------------------------------------- main


def floats(value: Any) -> Iterator[float]:
    """The floats in a detection result, in key order."""
    if isinstance(value, dict):
        for v in value.values():
            yield from floats(v)
    elif isinstance(value, float):
        yield value


def decision(router: Router, state: Any) -> Dict[str, str]:
    d = router.route(state)
    return {"model": d["model"], "reason": d["reason"]}


def main() -> None:
    fixtures = os.path.join(CRATE, "tests", "fixtures")
    os.makedirs(fixtures, exist_ok=True)

    ranges = class_ranges()
    write_tables(os.path.join(CRATE, "src", "tables.rs"), ranges)
    with open(os.path.join(fixtures, "chars.json"), "w") as f:
        json.dump(chars_fixture(ranges), f, separators=(",", ":"))

    with open(os.path.join(fixtures, "lang_code.jsonl"), "w") as f:
        for code in LANG_CODES:
            f.write(json.dumps({"code": code, "english": _english_from_code(code)}, ensure_ascii=False) + "\n")

    default, multilingual = Router(), Router(default="multilingual")
    path = os.path.join(fixtures, "route.jsonl")
    seen, stats = set(), Counter()
    with open(path, "w") as f:
        for sid, raw in all_states():
            assert sid not in seen, sid
            seen.add(sid)
            state = json.loads(json.dumps(raw, ensure_ascii=False))
            analysis = lang.analyse(state)
            rec = {
                "id": sid,
                "state": state,
                "analysis": analysis,
                "latin": lang.latin_profile(lang.state_text(state)),
                "route": decision(default, state),
            }
            route_ml = decision(multilingual, state)
            if route_ml != rec["route"]:
                rec["route_ml"] = route_ml
            rec["floats"] = [repr(x) for part in ("analysis", "latin") for x in floats(rec[part])]
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            reason = rec["route"]["reason"]
            stats["reason: " + re.sub(r"\d+%|\(.*?,|'\w+'", "…", reason)] += 1
            stats["script: %s" % analysis["script"]] += 1
            stats["language: %s" % analysis["language"]] += 1
    for k, v in sorted(stats.items()):
        print("%6d  %s" % (v, k))
    print("wrote %d states to %s (%.2f MB)" % (len(seen), path, os.path.getsize(path) / 2**20))


if __name__ == "__main__":
    main()
