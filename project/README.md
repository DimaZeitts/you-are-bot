# You Are Bot — определение бота в диалоге

Соревнование: <https://stackmorelayers.ai/competitions/you-are-bot>. Для каждого участника диалога
предсказывается вероятность того, что он бот; метрика — LogLoss.
Итог: CV **0.0990**, публичный лидерборд **0.0811**.

## Запуск end-to-end

```bash
# данные соревнования: train.json, ytrain.csv, test.json, ytest.csv -> data/raw/
./run.sh
```

`run.sh` создаёт `.venv`, ставит зависимости из `requirements.txt`, проверяет данные и запускает
эксперимент: признаки → кросс-валидация → обучение на всём train → сабмит `results/submission_v5.csv`.
Всё логируется в MLflow.

- без CV (только обучение и сабмит): `./run.sh --no-cv`;
- переопределить любой параметр: `./run.sh --set level2.n_seeds=3 cv.seeds=[0,1,2]`.

Нужен Python 3.14 (другой интерпретатор: `PYTHON=python3.14 ./run.sh`). Полный прогон на MacBook Air M2 —
~15 минут (`resources.threads: 4`).
При первом запуске с Hugging Face скачиваются модели эмбеддингов rubert-tiny2 и multilingual-e5-base.

## Трекинг (MLflow)

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db   # запускать из этой папки
```

Эксперимент `you-are-bot`, прогоны версий решения v2–v5:

| Прогон | CV LogLoss | LB LogLoss |
|---|---|---|
| v2_lgb_stack | 0.1143 | 0.0926 |
| v3_et_msglr | 0.1083 | 0.0834 |
| v4_blend_dialog | 0.1027 | 0.0811 |
| v5_bag_temp | 0.0990 | 0.0811 |

В каждом прогоне:
- **параметры** — все ключи `configs/config.yaml`;
- **метрики** — LogLoss каждого фолда, итоговый CV, время; `lb_logloss` — скор публичного лидерборда,
  дописан в прогон после отправки сабмита;
- **артефакты** — сабмит, OOF-предсказания, важность признаков, итоговый конфиг, `requirements.txt`;
- **теги** — MD5 файлов данных, версия Python, архитектура.

## Код

```
project/
├── run.sh               # точка входа
├── requirements.txt     # все зависимости с точными версиями
├── configs/config.yaml  # все параметры эксперимента
├── mlflow.db, mlruns/   # трекинг: прогоны v2–v5
└── src/youarebot/
    ├── data.py          # загрузка данных, таблица «один участник — одна строка», базовые признаки
    ├── features.py      # позиция во времени, стиль, эмбеддинги, kNN по репликам, таблица сообщений
    ├── models.py        # 1-й уровень (OOF), 2-й уровень (LightGBM), бленд, бэггинг, температура
    ├── pipeline.py      # CV → обучение → сабмит, логирование в MLflow
    └── config.py        # загрузка конфига, ограничение потоков
```
