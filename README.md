# mle-template-case-sprint2

Добро пожаловать в репозиторий-шаблон Практикума для проекта 2 спринта. Ваша цель — улучшить ключевые метрики модели для предсказания стоимости квартир Яндекс Недвижимости.

Полное описание проекта хранится в уроке «Проект. Улучшение baseline-модели» на учебной платформе.

Здесь укажите имя вашего бакета: ```s3-student-mle-20240328-6bcf522120```

Цель данного проекта - улучшение baseline модели, обученной в 1 спринте. В папке mlflow_server в файле notebook.ipynd происходит подключение к mlflow и настройка сервера, обучение baseline модели и ее сохранение в mlflow. В папке model_improvement в файле project_template_sprint_2.ipynb происходит анализ данных, улучшение модели и обучение.

Используемые технологии:

Visual Studio Code,
Jupyter Notebook,
база данных PostgreSQL с данными для обучения,
база данных PostgreSQL с данными для MLflow,
объектное хранилище для MLflow,
сервисы MLflow (Tracking Server и Model Registry),
нужные библиотеки для разработки библиотеки.

Клонирование проекта:
```
git clone https://github.com/tinanator/mle-project-sprint-2.git
cd mle-project-sprint-2
pip install -r requirements.txt 
```

# Руководство по проекту

**Этап 1: Разворачивание MLflow и регистрация модели**

Чтобы поднять mlflow сервис, из папки mlflow_server, нужно запустить команду ```sh run_mlflow_server.sh```. Скрипт лежит в файле mlflow_server/run_mlflow_server.sh. Перед этим экспортировать переменные среды ```export $(cat .env | xargs)```.

Имя эксперимента ```experiment_project_sprint_2_v2```. ID эксперимента 31.

Baseline модель сохранена в запуске basiline_model_registry_0. Название модели: baseline_flat_model.   

Результат r2 на тестовых данных 0.02120854295068797.

Этапы 2-5 находятся в ноутбуке model_improvement/project_template_sprint_2.

**Этап 2: Проведение EDA**

1. Проводим общий анализ датасета.
2. Анализ целевой переменной
3. Проводим анализ признаков и визулизацию признаков.
4. Анализ целевой переменной в зависимости от различных признаков.
5. Выводы:

    1. Квартиры до 1960 года в среднем дороже квартир после 1960 года. При этом преобладающее количество квартир было построено примерно после 1960 года.
    2. Чем выше этаж, на котором расположена квартира, тем в среднем она будет дороже.
    3. Стоимость квариры не сильно зависит от наличия лифта, но наличие лифта увеличивает стоимость.
    4. Квартиры, которые являются апартаментами, в среднем дороже квартир, не являющихся апартаментами.
    5. В центре плотность квартир и стоимость на них увеличивается.
    6. В среднем квартиры с большей площадью и жилой площадью дороже квартир с меньшей. Сложно отследить зависимость стоимости от площади кухни. Стоимость растет с увеличением высоты потолка до высоты со значением 5. После этого стоимость падает и остается примерно на одном уровне до высоты 25. 
    7. Стоимость так же зависит от типа квартиры.

Результаты сохранены в запуске eda_results_0.

**Этап 3: Генерация признаков и обучение модели**

Ручная генерация

К датасету были применены StandardScaler для числовых данных и OneHotEncoder для категориальных. В результате получились следующие дополнительные признаки: building_type_int_0, building_type_int_1,
       building_type_int_2, building_type_int_3, building_type_int_4,
       building_type_int_5, building_type_int_6, has_elevator_False,
       has_elevator_True, is_apartment_False, is_apartment_True.

После этого были применены KBinsDiscretizer для колонок ['latitude', 'longitude'] и PolynomialFeatures для ['total_area', 'living_area'].

В результате получены следующие признаки: KBinsDiscretizer__latitude_0.0,	KBinsDiscretizer__latitude_1.0,	KBinsDiscretizer__latitude_2.0,	KBinsDiscretizer__latitude_3.0,	KBinsDiscretizer__latitude_4.0,	KBinsDiscretizer__longitude_0.0,	KBinsDiscretizer__longitude_1.0,	KBinsDiscretizer__longitude_2.0,	KBinsDiscretizer__longitude_3.0,	KBinsDiscretizer__longitude_4.0,	PolynomialFeatures__total_area,	PolynomialFeatures__living_area,	PolynomialFeatures__total_area^2,	PolynomialFeatures__total_area, living_area,	PolynomialFeatures__living_area^2.



Далее была применена автоматическая генерация признаков с помощью AutoFeatRegressor для колонок ['total_area', 'ceiling_height']. Полученные признаки: total_area\*\*2,	ceiling_height\*\*2,	ceiling_height\*total_area,	ceiling_height\*total_area\*\*2.

Запускк в MLFLOW:

Feature_generation_0 - модели преобразования и генерации новых признаков.

model_2_registry_0 - логирование модели с метриками.

**Этап 4: Отбор признаков и обучение новой версии модели**

Для отбора признаков используем сначала метод SequentialFeatureSelector forward, отбираем 15 признаков. 
Отобранные признаки: 'KBinsDiscretizer__latitude_0.0' 'KBinsDiscretizer__latitude_3.0'
 'KBinsDiscretizer__latitude_4.0' 'KBinsDiscretizer__longitude_1.0'
 'KBinsDiscretizer__longitude_2.0' 'KBinsDiscretizer__longitude_3.0'
 'KBinsDiscretizer__longitude_4.0' 'PolynomialFeatures__total_area'
 'PolynomialFeatures__total_area^2' 'ceiling_height' 'building_type_int_1'
 'building_type_int_2' 'latitude' 'kitchen_area' 'floors_total'.

Далее отбираем признаки методом SequentialFeatureSelector backward, так же 15 признаков.
Отобранные признаки: 'KBinsDiscretizer__latitude_0.0' 'KBinsDiscretizer__latitude_2.0'
 'KBinsDiscretizer__latitude_4.0' 'KBinsDiscretizer__longitude_0.0'
 'KBinsDiscretizer__longitude_3.0' 'KBinsDiscretizer__longitude_4.0'
 'PolynomialFeatures__total_area' 'PolynomialFeatures__total_area^2'
 'ceiling_height' 'building_type_int_4' 'building_type_int_6'
 'has_elevator_True' 'latitude' 'kitchen_area' 'floors_total'.

 Объединяем списки, чтобы получить итоговый список признаков: 'building_type_int_1', 'KBinsDiscretizer__latitude_0.0', 'floors_total',
 'building_type_int_6', 'PolynomialFeatures__total_area^2', 'PolynomialFeatures__total_area', 'KBinsDiscretizer__longitude_2.0',
 'KBinsDiscretizer__longitude_3.0', 'building_type_int_2', 'KBinsDiscretizer__longitude_1.0',
 'KBinsDiscretizer__latitude_4.0', 'KBinsDiscretizer__longitude_0.0',
 'KBinsDiscretizer__latitude_2.0','KBinsDiscretizer__longitude_4.0', 'kitchen_area',
 'building_type_int_4', 'KBinsDiscretizer__latitude_3.0', 'ceiling_height', 'latitude',
 'has_elevator_True'.

 Запуски: 
 
 Feature_selection_0 - логгирование моделей отбора признаков и графики, визуализирующие отбор.
 
 model_3_registry_0 - логирование модели с метриками 

**Этап 5: Подбор гиперпараметров и обучение новой версии модели**

Отбираем гиперпараметры с помощью optuna. Берем следующую сетку параметров:
'learning_rate': [1e-4, 1e-1]
'depth': [6,10]
'iterations': [30, 100]

Для отбора лучших гиперпараметров используем 10 испытаний и минимизируем среднюю квавдратичную ошибку. Процесс отбора залогирован в запуске hyperparameters_selection_optuna_0. 
Лучшие гиперпараметры: {'learning_rate': 0.02325491762099814, 'depth': 10, 'iterations': 48}

Далее пробуем отобрать параметры с помощью GridSearch. Используем сетку:
    'depth' : [6,8,10],
    'learning_rate' : [1e-4, 1e-3, 1e-2, 1e-1],
    'iterations' : [30, 50, 100].
    
Лучшие гиперпараметры: {'depth': 10, 'iterations': 100, 'learning_rate': 0.1}

С помощью optuna построим график влияния гиперпараметров на модель. Больше всего влияние имеет learning_rate (0.75). Остальные параметры имеют небольшое влияние на результаты модели: depth - 0.14 и iterations - 0.11.

Будем обучать модель с гиперпараметрами, полученными optuna, так как результат средней квадратичной ошибки optuna (3181799.326083065) лучше, чем результат GridSearch (6993009109360.341).

Результат модели (метрика r2), обученной на лучших гиперпараметрах, составляет 0.5785755662605767. Видим, что результат значительно поднялся с 0.02120854295068797 до 0.5785755662605767.

Модель залогирована в запуске model_4_registry_0.

