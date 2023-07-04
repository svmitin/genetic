# Genetic

### Описание задачи
Нужно написать класс, принимающий на вход:\
    * M - целое число\
    * N - размер возвращаемого массива целых чисел\
    * количество особей в популяции\
    * максимальное количество поколений\
Класс должен найти массив чисел в количестве N, произведение которых совпадает с M или близко к нему\
Класс должен реализовать собой [генетический алгоритм](https://ru.wikipedia.org/wiki/Генетический_алгоритм)\
При этом первое число должно быть больше или равно последнему\
Класс должен возвращать:\
    * Найденный массив чисел\
    * Произведение чисел найденного массива\
    * Прочую сопутствующую информацию для анализа работы\

### Настройка окружения
```bash
python -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### Настройка СУБД PostgreSQL
При необходимости повторной тренировки ML-модели потребуется СУБД PostgreSQL. Для этого нужно создать пользователя PostgreSQL, делательно с логином и паролем `ml`. После этого нужно создать БД
```sql
CREATE DATABASE ml
    WITH
    OWNER = ml
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1
    IS_TEMPLATE = False;
```

После чего создать таблицу для этой БД
```sql
CREATE TABLE IF NOT EXISTS public.genetic
(
    id bigint NOT NULL DEFAULT nextval('genetic_id_seq'::regclass),
    target integer NOT NULL,
    properties_count integer NOT NULL,
    population_count integer NOT NULL,
    depth integer NOT NULL,
    difference numeric(10,5) NOT NULL,
    CONSTRAINT genetic_pkey PRIMARY KEY (id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.genetic
    OWNER to ml;
```

Скрипт `ml.py` реализующий работу с данной БД имеет предустановленные параметры подключения для удобства. Именно по этому имя пользователя, его пароль и имя БД должны иметь значение `ml`, но это не обязательно. В качестве СУБД выбрана именно PostgreSQL для возможности удаленного подключения к БД и соответственно майнинга данных с множества ПК/серверов

### Запуск приложения genetic.py
Обычный Python3 скрипт реализующий генетический алгоритм
Описание данной программы можно увидеть выше в графе "Описание задачи"
```bash
python3 genetic.py --target=225792 --properties_count=12 --population_count=100 --depth=10
```
Также возможен запуск без указания количества особей в популяции и максимального количества поколений. В таком случае данные параметры будут автоматически выставлены обученной ML-моделью. Программа вернет результат быстрее и качественней, так как параметры будут оптимальны, но фактическое время работы (возможно) станет больше, в виду долгой загрузки ML-модели.
```bash
python3 genetic.py --target=225792 --properties_count=12
```
Пример вывода на экран:
```
Заданная цель: 225792
Заданное количество чисел в массиве: 12
Заданное количество особей в популяции: 52
Заданное количество поколений: 58
Результат: 225792
Разница: 0.0%
Найденный массив чисел: 3, 4, 4, 4, 1, 1, 2, 7, 4, 7, 1, 3
Количество прошедших поколений: 8
```

### Майнинг данных для обучения нейронной модели
Вызывается только метод `mining_data()`, работающий бесконечно. Он запускает предыдущую программу `genetic.py` с частично случайными параметрами. Сохраняет в БД параметры вызова `genetic.py` и % отклонения результата от цели.
Требует создания БД PostgreSQL.
```bash
python3 ml.py --task=mining --dbuser=ml --dbpass=ml --dbhost=localhost --dbport=5432 --dbname=ml
```
Приведенные в примере параметры запуска данного скрипта аналогичны параметрам по умолчанию, что позволяет запускать скрипт без параметров вовсе:
```bash
python3 ml.py
```

### Обучение нейронной модели
Вызывает метод `create_model()` для создания и тренировки модели. В результате создается файл `model.h5`.
```bash
python3 ml.py --task=train
```
После успешного обучения ML-модели можно повторно запускать скрипт `genetic.py`. За определение оптимального количества особей в популяции и максимального количества поколений будет отвечать вновь обученная ML-модель. Она будет загружена из файла `model.h5`
```bash
python3 genetic.py --target=225792 --properties_count=12
```
