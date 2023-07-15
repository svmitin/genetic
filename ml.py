from random import randint, choice
from math import prod
from datetime import datetime
import multiprocessing

import click
from sqlalchemy import create_engine, MetaData
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean
from sqlalchemy.orm import Session, declarative_base
from sqlalchemy import distinct
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils
import numpy as np

from genetic import Farm


DBUSER = 'ml'
DBPASS = 'ml'
DBHOST = 'localhost'
DBPORT = '5432'
DBNAME = 'ml'

Base = declarative_base()
engine = create_engine(f'postgresql+psycopg2://{DBUSER}:{DBPASS}@{DBHOST}:{DBPORT}/{DBNAME}')
metadata = MetaData()
session = Session(bind=engine)


PROPERTIES = [
    [4, 7, 10],
    [5, 8, 11],
    [6, 9, 12],
    [7, 10, 13],
    [8, 11, 14],
]


class GeneticData(Base):
    __tablename__ = 'genetic'
    id = Column(Integer, primary_key=True, autoincrement=True)
    target = Column(Integer)
    properties_count = Column(Integer)
    population_count = Column(Integer)
    depth = Column(Integer)
    difference = Column(Float)

    def __init__(self, target, population_count, properties_count, depth, difference) -> None:
        self.target = target
        self.properties_count = properties_count
        self.population_count = population_count
        self.depth = depth
        self.difference = difference


def new_session():
    '''Пересоздает подключение к БД'''
    global session
    session = Session(bind=engine)


def get_difference(target, properties_count, population_count, depth):
    '''возвращает средние значения по глубине и % разницы для заданных аргументов'''
    differences, depths = [], []
    for _ in range(5):
        result = Farm(
            target,
            properties_count,
            population_count,
            depth
        ).run()
        differences.append(result['difference'])
        depths.append(result['depth'])
    return int(sum(depths)/len(depths)), round(sum(differences)/len(differences), 5)


def mining_data():
    '''бесконечно добывает данные для обучения модели'''
    while True:
        for properties_count in choice(PROPERTIES):                          # ml input
            target = prod([randint(1, 8) for t in range(properties_count)])  # ml input
            found = False

            prev_avg_difference = 0
            result_does_not_change_count = 0
            for depth in range(100, 700, 100):                              # ml output. DONT CHANGE
                for population_count in range(20, 900, 10):                 # ml output
                    avg_depth, avg_difference = get_difference(target, properties_count, population_count, depth)
                    # print(f'target: {target} properties_count: {properties_count} population_count: {population_count} avg_depth: {avg_depth} avg_difference: {avg_difference}')
                    data = GeneticData(
                        target, 
                        population_count,
                        properties_count,
                        avg_depth,
                        avg_difference
                    )
                    if avg_difference > 7:
                        break
                    if prev_avg_difference == avg_difference:
                        result_does_not_change_count += 1
                    if avg_difference < 0.7:
                        session.add(data)
                    if avg_difference < 0.4 or result_does_not_change_count > 5:
                        found = True
                        break
                if found:
                    break
        try:
            session.commit()
            print(f'save data')
        except:
            new_session()
            print(f'reconnected')


def load_data():
    '''подготовка данных. отсивается лишнее, на лучших значениях будет обучаться модель'''
    raw_data = session.query(GeneticData).all()
    data = {}
    for row in raw_data:
        target = row.target
        properties_count = row.properties_count

        data.setdefault(target, {})
        data[target].setdefault(properties_count, {
            'difference': 100,
            'population_count': -1,
            'depth': -1,
        })
        if row.difference < data[target][properties_count]['difference']:
            data[target][properties_count] = {
                'difference': row.difference,
                'population_count': row.population_count,
                'depth': row.depth,
            }
    
    prepared_data = []
    for target in data.keys():
        for properties_count in data[target].keys():
            if data[target][properties_count]['difference'] < 3:
                prepared_data.append({
                    'target': target,
                    'properties_count': properties_count,
                    'population_count': data[target][properties_count]['population_count'],
                    'depth': data[target][properties_count]['depth'],
                    'difference': data[target][properties_count]['difference']
                })
    
    x_train, y_train, x_test, y_test = [], [], [], []
    for i, line in enumerate(prepared_data):
        x = [
            line['target'],
            line['properties_count'],
        ]
        y = [
            line['population_count'],
            line['depth'],
        ]
        if i % 30 == 0:
            x_test.append(x)
            y_test.append(y)
            continue
        x_train.append(x)
        y_train.append(y)
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, batch_size=300, epochs=1200):
    '''создает, обучает и возвращает регрессионную модель'''
    model = Sequential([
        Dense(2, activation='relu'),
        Dense(6),
        Dense(3),
        Dense(2)
    ])
    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['mae']
    )
    model.fit(
        np.array(x_train),          # данные для обучения
        np.array(y_train),          # правильные ответы 
        batch_size=batch_size,      # размер мини-выборки
        epochs=epochs,              # количество эпох обучения
        validation_split=0.2,
        verbose=0
    )
    return model


def train_model(params):
    '''Тренирует модель. Обычно вызывается в отдельном процессе'''
    batch_size, epochs, x_train, y_train, x_test, y_test = params
    print(f'start train with batch_size: {batch_size}, epochs: {epochs}')
    model = create_model(x_train, y_train, batch_size, epochs)
    print(f'end train with batch_size: {batch_size}, epochs: {epochs}')
    differences = []
    
    predictions = model.predict(np.array([x_test, y_test]))
    for i, prediction in enumerate(predictions[0]):
        target = x_test[i][0]
        properties_count = x_test[i][1]
        try:
            population_count = int(prediction[0])
            depth = int(prediction[1])
            print(f'target: {target} properties_count: {properties_count} population_count: {population_count} depth: {depth}')
            avg_depth, avg_difference = get_difference(target, properties_count, population_count, depth)
            differences.append(avg_difference)
        except:
            print('error')
            population_count = 1000
            depth = 1000
            avg_depth = 1000
            avg_difference = 100
            differences.append(avg_difference)

    result = {
        'avg_difference': round(sum(differences) / len(differences), 5),
        'batch_size': batch_size,
        'epochs': epochs
    }
    return result


def autotrain():
    '''Самостоятельно пытается обучить нейронную сеть'''
    x_train, y_train, x_test, y_test = load_data()
    params = []
    for batch_size in [350]:
        for epochs in range(3200, 3800, 100):
            params.append((batch_size, epochs, x_train, y_train, x_test, y_test,))
    
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_count)
    good_params = pool.map(train_model, params)

    good_params = {gp['avg_difference']: gp for gp in good_params}
    if good_params:
        best_difference = min(good_params)
        best_params = good_params[best_difference]
        print(best_params)
        model = create_model(x_train, y_train, best_params['batch_size'], best_params['epochs'])
        model.save('model.h5')
    print(good_params)


@click.command()
@click.option('--task', default='mining', help='Выполняемая операция. Варианты: mining - майнинг данных; train - автоматическая тренировка и сохранение модели')
@click.option('--dbuser', default='ml', help='Имя пользователя для подключения к БД')
@click.option('--dbpass', default='ml', help='Пароль для подключения к БД')
@click.option('--dbhost', default='localhost', help='Хост БД')
@click.option('--dbport', default=5432, help='Порт БД')
@click.option('--dbname', default='ml', help='Имя БД')
def run(task, dbuser, dbpass, dbhost, dbport, dbname):
    DBUSER = dbuser
    DBPASS = dbpass
    DBHOST = dbhost
    DBPORT = dbport
    DBNAME = dbname
    new_session()
    if task == 'mining':
        mining_data()
    if task == 'train':
        autotrain()


if __name__ == '__main__':
    run()
