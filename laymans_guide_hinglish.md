# AeraQuant Layman's Guide (Hinglish Version)

Is guide mein hum AeraQuant ko ekdum simple Hinglish mein samjhenge taaki ek 14 saal ke bachhe ko bhi samajh aa jaye ki ye project kya karta hai aur iske peeche kya science hai.

---

## 1. AeraQuant kya hai? (In Simple Terms)

AeraQuant ek Smart Trading Analytics System hai.
Jaise bade banks (JPMorgan, Goldman Sachs) ya proprietary trading firms ke paas trading screens hoti hain jo unhe batati hain ki kaun sa asset (Gold, Crude Oil, Dollar) sasta hai aur kaun sa mehenga, ye dashboard wahi kaam karta hai.

Ye market se live data khinchta hai, mathematical formulas lagata hai, aur batata hai ki kya trade lena chahiye aur kab market dangerous hai.

---

## 2. Pepsi-Coke Analogy (Dono Assets ka Rishta)

Pehle ek simple example lete hain. Maan lo market mein do Cold Drinks hain: **Pepsi** aur **Coke**.

1. Dono ki price lagbhag barabar rehti hai (maan lo ₹40 aur ₹42).
2. Achanak kya hota hai, Coke ki price badh ke ₹60 ho jati hai, par Pepsi ₹40 par hi rehti hai.
3. Hum sabko pata hai ki dono lagbhag same hain, toh dono ki price fir se barabar aayegi.
4. **Hamara Trade:** Hum mehenge Coke ko sell (short) karenge aur saste Pepsi ko buy karenge.
5. Jab dono wapas ₹45-₹45 par aa jayenge, hum apna trade close karke profit book kar lenge.

Is strategy ko quantitative trading mein **Pairs Trading (ya Mean Reversion)** kehte hain. Aur hamare project ka main focus yahi check karna hai.

---

## 3. Dashboard ke main parts kya hain?

### A. Data Pipelines (Yaani Dukan se samaan lana) 

- **Code:** `data/fetcher.py`
- **Kaam:** Ye software automatically **Yahoo Finance** ke server par jata hai aur hume pichle **5 saal (approx. 1,260 days)** ke daily prices laa ke deta hai.
- **12 Assets:** Hum total 12 alag-alag cheezon ko track karte hain — Jaise Energy (Oil, Gas), Metals (Gold, Silver, Copper), Equity Indices (S&P 500, Nasdaq), Rates, aur Foreign Exchange currencies (EUR, GBP).

### B. Cointegrated Spreads & Z-Score (Dono ke price ka gap measure karna)

- **Code:** `analysis/spreads.py`
- **Kaam:** Hum pairs banate hain (jaise Gold aur Silver). Hum inke price ka ratio nikalte hain.
- **Z-Score kya hai?** Z-Score hume batata hai ki dono prices ka gap normal se kitna door chala gaya hai.
  - Agar **Z-Score > +2** hai, iska matlab ratio bahut badh gaya hai (Pepsi relative to Coke bahut mehengi ho gayi hai). **SELL SIGNAL!**
  - Agar **Z-Score < -2** hai, iska matlab ratio bahut gir gaya hai. **BUY SIGNAL!**

### C. GARCH(1,1) Volatility Model (Risk calculate karna)

- **Code:** `analysis/indicators.py`
- **Kaam:** Volatility ka matlab hota hai market mein kitna jhatka (ups & downs) lag raha hai.
- GARCH ek statistical model hai jo ye dekhta hai ki market mein abhi risk high hai ya low. Agar market mein halchal bahut zyada hai (High Regime), toh hum trade karne se bachte hain kyunki wahan pairs ka rishta toot sakta hai.
- Hum 5 saal ka data isliye use karte hain kyunki GARCH ko accurate hone ke liye kam se kam 1,000 din ka data chahiye hota hai.

### D. VIX Global Risk Overlay (Pure market ka dar napna)

- **VIX (CBOE Volatility Index):** Isko **Fear Index** bhi bolte hain.
- Jab pure global market mein panic hota hai (jaise COVID ya recession), toh VIX 25 se upar chala jata hai.
- **Smart Rule:** Agar VIX > 25 hai (yani pure market mein darr ka mahaul hai), toh hamara system saare **BUY** signals ko block karke **HOLD** pe daal deta hai, kyunki jab sab kuch toot raha ho toh cointegration fail ho sakti hai.

### E. Strategy Backtester (Time Travel trading simulation)

- **Code:** `analysis/backtester.py`
- **Kaam:** Agar hum pichle 5 saal mein Z-Score ke rule par trading karte, toh kya sach mein hume profit hota ya loss?
- Ye simulator pichle 5 saal ke pure data par trade execute karke equity curve (profit growth chart) banata hai.
- Ye hume performance metrics batata hai:
  - **Sharpe Ratio:** Ye check karta hai ki risk ke mukable kitna return mila. (Sharpe > 1.0 matlab good strategy, > 2.0 matlab top-tier hedge fund level).
  - **Max Drawdown:** Strategy se pichle 5 saal mein sabse bada continuous loss percentage kitna hua.
  - **Win Rate:** Kitne percent trades profit mein close hue.

---

## 4. Recruiter (Futures First) ko kya impact dikhega?

Recruiter jab tumhara CV dekhega toh use samajh aayega ki:

1. Tumhe sirf trading aati nahi hai, tumne use program kiya hai.
2. Tumhe financial mathematics (GARCH, Cointegration) ke practical applications pata hain.
3. Tumne real markets (Exchange data) ko test karne ke liye khud ka backtester banaya hai.
4. Tum capital protect karna jaante ho (VIX and Volatility filters ke through).
