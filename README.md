# Magnit Uplift Modeling

**3-е место** на хакатоне Magnit Tech x HSE 2026 — «Машинное обучение для ритейла», трек «Uplift-моделирование».

Ключевая идея решения: ансамбль двух X-Learner моделей на LightGBM — глобальной (на всей выборке) и per-channel (отдельная модель для каждого канала коммуникации), объединённых ранговым блендингом.

---

## Задача

Предсказать гетерогенный эффект маркетинговых коммуникаций на траты покупателя (`rec_spend`) — то есть оценить **CATE (Conditional Average Treatment Effect)** для каждого клиента. Выборка содержит три канала коммуникаций (`com_type_1/2/3`), ~355K пользователей в трейне, ~118K в тесте. Задача — отранжировать аудиторию по «отзывчивости» на коммуникацию.

Метрика: **uplift@10** в рублях — разница средних трат между обработанными и контрольными клиентами в топ-10% по предсказанному uplift.

---

## Данные

| | Строк | Целевая переменная |
|---|---|---|
| `train.parquet` | 355 246 | `rec_spend` (непрерывная, 90.2% нулей) |
| `test.parquet` | 118 414 | — |

- `treatment_flg` — флаг воздействия (49.6% treated, практически идеальный RCT)
- `communication_type` — канал коммуникации (3 типа, ~118K пользователей каждый)
- 88 признаков: демография, RFM-метрики, категориальная аффинность (cat5/6/7), маркетинговая история

**Гетерогенность по каналам** (средний uplift в рублях):

| Канал | Uplift |
|---|---|
| com_type_3 | +4.37 ₽ |
| com_type_1 | +3.78 ₽ |
| com_type_2 | +1.42 ₽ |

---

## Решение

### Финальная модель

**X-Learner на LightGBM** — двухстадийный мета-алгоритм оценки CATE.

Алгоритм:
1. **Stage 1**: Обучить `μ₁(X)` на treated, `μ₀(X)` на control (регрессия LightGBM)
2. **Stage 2**: Вычислить imputed эффекты `D₁ = Y₁ − μ₀(X₁)` и `D₀ = μ₁(X₀) − Y₀`, обучить `τ₁`, `τ₀` на них
3. **CATE**: `g(X) · τ₀(X) + (1 − g(X)) · τ₁(X)`, где `g(X)` — propensity score (логистическая регрессия)

**Ансамбль**:
- Глобальный X-Learner (все данные)
- Per-channel X-Learner (три отдельные модели по каналам)
- 5-seed усреднение для каждого
- Ранговый блендинг: `0.7 × perchan_ranks + 0.3 × global_ranks`

Гиперпараметры подобраны через **Optuna** (40-80 trials на 40% подвыборке).

### Инженерия признаков

10 дополнительных признаков поверх 88 сырых:

| Признак | Формула |
|---|---|
| `cat7_share` | `cus_cat_7_rto / rto` |
| `mkt_resp_rate` | `cus_mark_n_rule / cus_mark_n_offers` |
| `mkt_view_rate` | `cus_mark_n_view / cus_mark_n_offers` |
| `spend_cv` | `stdtv / atv` (коэффициент вариации трат) |
| `trn_density` | `n_trn / n_days_life × 365` (транзакции/год) |
| `recency_ratio` | `cus_cat_7_last_1_days / avg_days_btw_trn` |
| `cat_breadth_ratio` | `n_cat_7 / n_cat_5` |
| + `cat6_share`, `cat5_share`, `cat7_vs_last_visit` | аналогичные доли |

### Adversarial Validation

Обнаружен сдвиг распределения train→test (AUC = 0.888). Топ смещённых признаков: `n_days_life`, `cus_mark_n_offers`, `spend_cv`, `trn_density`. Учтено при выборе фичей для v2-версии модели.

---

## Результаты бенчмарка

OOF **uplift@10 в рублях** (5-fold StratifiedKFold по `treatment × channel`):

| # | Модель | uplift@10 (₽) | CI 80% |
|---|---|---|---|
| 1 | **X-Learner Optuna (global)** | **21.07** | [19.49, 22.45] |
| 2 | Per-Channel X-Learner Optuna | 20.61 | — |
| 3 | **Финальный бленд (0.7/0.3)** | **21.24** | — |
| 4 | X-Learner (default) | 19.96 | [18.25, 21.89] |
| 5 | T-Learner | 18.50 | [16.84, 20.51] |
| 6 | Optuna Hurdle T-Learner | 18.05 | [17.06, 19.82] |
| 7 | DR-Learner | 17.05 | [15.40, 18.98] |
| 8 | Hurdle T-Learner | 16.75 | [15.65, 18.70] |
| 9 | Per-Channel Hurdle | 15.08 | [13.76, 16.93] |

### Deep Learning бенчмарк

Дополнительно проверены нейросетевые архитектуры (PyTorch, Apple Silicon MPS):

| Модель | uplift@10 (бинарный) |
|---|---|
| TARNet | 20.50 |
| CFR (Wasserstein) | 20.17 |
| CFR (MMD) | 19.84 |
| DragonNet | 19.56 |
| RERUM | 17.91 |

---

## Структура репозитория

```
magnit_uplift/
├── EDA.ipynb                        # Разведочный анализ, гипотезы, baseline T-Learner
├── pilot_modeling.ipynb             # Бенчмарк 9 моделей, Optuna, adversarial val
├── pilot_modeling_final.ipynb       # Финальный пайплайн: обучение и сабмит
├── dl_uplift_benchmark.ipynb        # DL-бенчмарк (TARNet, CFR, DragonNet, RERUM)
│
├── train.parquet                    # Обучающая выборка (355K строк)
├── test.parquet                     # Тестовая выборка (118K строк)
├── data_description.csv             # Описание признаков
├── requirements.txt
│
├── pilot_artifacts/
│   ├── optuna_xl_params.json        # Optuna-параметры глобального X-Learner
│   ├── perchan_xl_params.json       # Параметры per-channel X-Learner (3 канала)
│   ├── optuna_xl_v2_params.json     # v2: overlap-валидация, top25 фичи
│   ├── optuna_xl_v3_params.json     # v3: tweedie + MAE, overlap-веса
│   ├── model_comparison.csv         # Таблица результатов (рублёвая шкала)
│   ├── pilot_summary.json           # Все OOF-метрики
│   ├── submission_blend_perchan07_global03_5seed.csv  # ФИНАЛЬНЫЙ САБМИТ
│   ├── submission_xl_opt_5seed.csv  # Глобальный X-Learner, 5 сидов
│   ├── submission_perchan_xl_opt_5seed.csv            # Per-channel, 5 сидов
│   ├── test_global_xl_5seed.npy     # Raw-предсказания глобальной модели
│   ├── test_perchan_xl_5seed.npy    # Raw-предсказания per-channel модели
│   ├── oof_*.npy                    # OOF-предсказания всех моделей
│   ├── model_comparison_plot.png
│   └── adversarial/                 # Adversarial validation выходы
│
├── model_artifacts/
│   ├── global_xlearner.pkl          # Сохранённый глобальный X-Learner
│   ├── perchan_xlearner.pkl         # Сохранённый per-channel X-Learner
│   ├── all_feats.json               # Список всех 96 признаков
│   ├── comm_codes.json
│   └── le_map.pkl
│
├── eda_artifacts/                   # EDA-артефакты: JSON, CSV, PNG
└── dl_artifacts/                    # DL-бенчмарк: OOF .npy, сравнение
```

---

## Воспроизведение

```bash
pip install -r requirements.txt
```

Запускать ноутбуки в порядке:

1. `EDA.ipynb` — генерирует `eda_artifacts/`
2. `pilot_modeling.ipynb` — бенчмарк, Optuna, генерирует `pilot_artifacts/`
3. `pilot_modeling_final.ipynb` — финальное обучение и сабмит (загружает сохранённые params)
4. `dl_uplift_benchmark.ipynb` — независимый DL-бенчмарк

Для воспроизведения финального сабмита достаточно запустить `pilot_modeling_final.ipynb` — он загружает готовые параметры из `pilot_artifacts/optuna_xl_params.json` и `pilot_artifacts/perchan_xl_params.json`.

---

## Стек

`lightgbm` · `causalml` · `econml` · `scikit-uplift` · `optuna` · `pytorch` · `shap` · `pandas` · `scikit-learn`
