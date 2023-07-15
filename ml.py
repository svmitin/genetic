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


Base = declarative_base()

PROPERTIES = [
    [4, 7, 10],
    [5, 8, 11],
    [6, 9, 12],
    [7, 10, 13],
    [8, 11, 14],
]

DBUSER = 'ml'
DBPASS = 'ml'
DBHOST = 'localhost'
DBPORT = '5432'
DBNAME = 'ml'


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


def train(params):
    batch_size, epochs = params
    designer = ModelDesigner(task=None)
    print('load_data')
    x_train, y_train, x_test, y_test = designer.load_data()
    print('start train')
    model = designer.create_model(x_train, y_train, batch_size, epochs)
    print('end train')
    differences = []
    
    predictions = model.predict(np.array([x_test, y_test]))
    for i, prediction in enumerate(predictions[0]):
        target = x_test[i][0]
        properties_count = x_test[i][1]
        try:
            population_count = int(prediction[0])
            depth = int(prediction[1])
            print(f'target: {target} properties_count: {properties_count} population_count: {population_count} depth: {depth}')
            avg_depth, avg_difference = designer.get_difference(target, properties_count, population_count, depth)
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

class ModelDesigner:
    def __init__(self, task='mining'):
        self.name = f'{datetime.now().second}'
        self.engine = create_engine(f'postgresql+psycopg2://{DBUSER}:{DBPASS}@{DBHOST}:{DBPORT}/{DBNAME}')
        metadata = MetaData()
        self.session = Session(bind=self.engine)
        if task == 'mining':
            self.mining_data()
        if task == 'train':
            self.autotrain()

    def get_difference(self, target, properties_count, population_count, depth):
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

    def mining_data(self):
        '''бесконечно добывает данные для обучения модели'''
        while True:
            for properties_count in choice(PROPERTIES):                          # ml input
                target = prod([randint(1, 8) for t in range(properties_count)])  # ml input
                found = False

                prev_avg_difference = 0
                result_does_not_change_count = 0
                for depth in range(100, 700, 100):                              # ml output. DONT CHANGE
                    for population_count in range(20, 900, 10):                 # ml output
                        avg_depth, avg_difference = self.get_difference(target, properties_count, population_count, depth)
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
                            self.session.add(data)
                        if avg_difference < 0.4 or result_does_not_change_count > 5:
                            found = True
                            break
                    if found:
                        break
            try:
                self.session.commit()
                print(f'{self.name}: save data')
            except:
                self.session = Session(bind=self.engine)
                print(f'{self.name}: reconnect')

    def load_data(self):
        '''подготовка данных. отсивается лишнее, на лучших значениях будет обучаться модель'''
        raw_data = self.session.query(GeneticData).all()
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

    def create_model(self, x_train, y_train, batch_size=300, epochs=1200):
        '''создает регрессионную модель'''
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
            epochs=epochs,
            validation_split=0.2,
            verbose=0
        )
        return model

    def train(batch_size, epochs):
        x_train, y_train, x_test, y_test = self.load_data()
        model = self.create_model(x_train, y_train, batch_size, epochs)
        differences = []
        for i, x in enumerate(x_test):
            target, properties_count = x, y_test[i]
            prediction = model.predict(np.array([[target, properties_count]]))
            population_count = int(prediction[0][0])
            depth = int(prediction[0][1])
            avg_depth, avg_difference = self.get_difference(target, properties_count, population_count, depth)
            differences.append(avg_difference)
        return {
            'avg_difference': round(sum(differences) / len(differences), 5),
            'batch_size': batch_size,
            'epochs': epochs
        }

    def autotrain(self):
        '''Самостоятельно пытается обучить нейронную сеть'''
        good_params = {}

        params = []
        for batch_size in [350]:
            for epochs in range(3200, 3800, 100):
                params.append((batch_size, epochs,))
        
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        good_params = pool.map(train, params)
        good_params = {gp['avg_difference']: gp for gp in good_params}
        if good_params:
            best_difference = min(good_params)
            best_params = good_params[best_difference]
            print(best_params)
            x_train, y_train, x_test, y_test = self.load_data()
            model = self.create_model(x_train, y_train, best_params['batch_size'], best_params['epochs'])
            model.save('model.h5')
        print(good_params)

        # prediction = model.predict(np.array([[target, properties_count]]))
        # population_count = int(prediction[0][0])
        # depth = int(prediction[0][1])
        # print(population_count, depth)


@click.command()
@click.option('--task', default='mining', help='Выполняемая операция. Варианты: mining - майнинг данных; train - тренировка и сохранение модели')
@click.option('--dbuser', default='ml', help='Произведение всех параметров должно быть близким к этому числу')
@click.option('--dbpass', default='ml', help='Количество параметров')
@click.option('--dbhost', default='localhost', help='Количество особей в популяции')
@click.option('--dbport', default=5432, help='Максимальное количество поколений (итераций отбора/скрещивания)')
@click.option('--dbname', default='ml', help='Максимальное количество поколений (итераций отбора/скрещивания)')
def run(task, dbuser, dbpass, dbhost, dbport, dbname):
    DBUSER = dbuser
    DBPASS = dbpass
    DBHOST = dbhost
    DBPORT = dbport
    DBNAME = dbname
    designer = ModelDesigner(task)


if __name__ == '__main__':
    run()
