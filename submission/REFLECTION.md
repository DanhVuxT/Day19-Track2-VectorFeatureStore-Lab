# Reflection - Lab 19

Path da chay: Lite (fastembed + Qdrant in-memory + SQLite Feast + FastAPI).

Bonus: da them `bonus/ARCHITECTURE.md`, `bonus/agent.py`, va `bonus/demo.py`; demo chay bang `python bonus/demo.py`.

Lab nay giup minh thay Vector Store khong chi la noi luu embedding, ma con la mot retrieval index can quan tam den model embedding, payload, top-k, latency va cach danh gia bang golden set. Hybrid search tot hon keyword hoac vector rieng le vi no ket hop duoc exact-match signal cua BM25 voi semantic signal cua vector; tren lab, hybrid thang trung binh va dac biet tot o mixed queries, noi query vua co tu khoa ky thuat vua co y dien giai.

Feature Store giup tranh loi training-serving skew, data leakage trong point-in-time join, va viec moi service tu tinh feature theo cach khac nhau. Kho khan chinh la moi truong Windows khong co bash/make, PATH khong thay uvicorn/feast trong notebook, va latency NB3 bi anh huong boi cold cache. Cach xu ly la dung venv Python 3.12, goi CLI bang duong dan venv, bat UTF-8/Jupyter dirs trong repo, them warmup va cache query embedding.
