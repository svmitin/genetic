from random import randint, choice
from math import prod
from datetime import datetime

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

class ModelDesigner:
    def __init__(self, task='mining', dbuser='ml', dbpass='ml', dbhost='localhost', dbport=5432, dbname='ml'):
        self.name = f'{datetime.now().second}'
        self.engine = create_engine(f'postgresql+psycopg2://{dbuser}:{dbpass}@{dbhost}:{dbport}/{dbname}')
        metadata = MetaData()
        self.session = Session(bind=self.engine)
        if task == 'mining':
            self.mining_data()
        elif task == 'train':
            self.autotrain()
        else:
            print(f'Команда не корректна: {task}')

    def __get_difference(self, target, properties_count, population_count, depth):
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
                        avg_depth, avg_difference = self.__get_difference(target, properties_count, population_count, depth)
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

    def __load_data(self):
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

    def create_model(self, batch_size=300, epochs=1200):
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
        x_train, y_train, x_test, y_test = self.__load_data()
        model.fit(
            np.array(x_train),  # данные для обучения
            np.array(y_train),  # правильные ответы 
            batch_size=batch_size,     # размер мини-выборки
            epochs=epochs,
            validation_split=0.2,
            # verbose=1
        )
        return model
    
    def autotrain(self):
        '''Самостоятельно пытается обучить нейронную сеть'''
        good_params = {}
        target = 225792
        properties_count = 12
        for batch_size in range(50, 400, 100):
            for epochs in range(2000, 4000, 100):
                model = self.create_model(batch_size, epochs)
                prediction = model.predict(np.array([[target, properties_count]]))
                population_count = int(prediction[0][0])
                depth = int(prediction[0][1])
                if population_count < 40 or depth < 1:
                    continue
                avg_depth, avg_difference = self.__get_difference(target, properties_count, population_count, depth)
                good_params[avg_difference] = {
                    'batch_size': batch_size,
                    'epochs': epochs
                }
        if good_params:
            best_difference = min(good_params.keys())
            best_params = good_params[best_difference]
            model = self.create_model(best_params['batch_size'], best_params['epochs'])
            model.save('model.h5')
            print(best_params)
        print(good_params)

        prediction = model.predict(np.array([[target, properties_count]]))
        population_count = int(prediction[0][0])
        depth = int(prediction[0][1])
        print(population_count, depth)


@click.command()
@click.option('--task', default='mining', help='Выполняемая операция. Варианты: mining - майнинг данных; train - тренировка и сохранение модели')
@click.option('--dbuser', default='ml', help='Произведение всех параметров должно быть близким к этому числу')
@click.option('--dbpass', default='ml', help='Количество параметров')
@click.option('--dbhost', default='localhost', help='Количество особей в популяции')
@click.option('--dbport', default=5432, help='Максимальное количество поколений (итераций отбора/скрещивания)')
@click.option('--dbname', default='ml', help='Максимальное количество поколений (итераций отбора/скрещивания)')
def run(task, dbuser, dbpass, dbhost, dbport, dbname):
    designer = ModelDesigner(task, dbuser, dbpass, dbhost, dbport, dbname)


if __name__ == '__main__':
    run()
