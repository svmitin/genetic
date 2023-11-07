from random import randint
from random import choice
from math import prod
from typing import Dict

import click
from tensorflow.keras.models import load_model
import numpy as np


class Person:
    def __init__(self, properties: list = None):
        self.properties = properties
        self.prod = prod(self.properties)

    def __mul__(self, other):  # оператор *
        '''скрещивание особи с другой порождает новую'''
        # для получения свойств новой особи возьмем по половине от текущей и другой особи
        median = int(len(self.properties) / 2)
        new_propeties = self.properties[:median] + other.properties[median:]

        # мутация: случайным образом изменим одно из свойств
        index = randint(0, len(self.properties) - 1)
        new_propeties[index] = randint(1, max(self.properties))

        return Person(new_propeties)


class Farm:
    def __init__(self, target: int, properties_count: int, population_count: int, depth: int):
        self.target = target
        self.properties_count = properties_count
        self.population_count = population_count
        self.depth = depth
        self.property_max = 8
        self.best_person = None

        # создание популяции
        self.population = []
        for _ in range(self.population_count):
            self.population.append(Person([randint(1, self.property_max) for i in range(self.properties_count)]))

    def __key(self, person: Person) -> int:
        '''используется при сортировке особей. отражает на сколько особь близка к идеалу'''
        result = self.target - person.prod
        result = result if result > 0 else result * -1
        if person.properties[0] < person.properties[-1]:
            return result * self.properties_count * self.population_count
        return result
    
    def run(self) -> Dict:
        '''запуск поиска лучшей особи'''
        iterations_count = 0
        for i, _ in enumerate(range(self.depth), start=1):
            iterations_count = i
            self.selection()
            self.pairing()
            if self.best_person:
                # нашлась особь с полным совпадением раньше времени
                break

        if not self.best_person:
            # идеальная особь не была встречена, ищем лучшего кандидата
            self.population.sort(key=lambda person: self.__key(person), reverse=True)
            self.best_person = self.population.pop()

        # найдем разницу в процентах
        if self.target >= self.best_person.prod:
            difference = (self.target / self.best_person.prod -1) * 100
        else:
            difference = (self.best_person.prod / self.target -1) * 100
        return {
            'Заданная цель': self.target,
            'Заданное количество чисел в массиве': self.properties_count,
            'Заданное количество особей в популяции': self.population_count,
            'Заданное количество поколений': self.depth,
            'Результат': self.best_person.prod,
            'Разница': f'{round(difference, 4)}%',
            'Найденный массив чисел': ', '.join(map(str, self.best_person.properties)),
            'Количество прошедших поколений': iterations_count,
            'depth': iterations_count,
            'difference': round(difference, 8)
        }

    def selection(self):
        '''отбор 20% самых подходящих особей'''
        new_population_count = int(self.population_count / 100 * 20)
        self.population.sort(key=lambda person: self.__key(person))
        self.population = self.population[:new_population_count]

    def pairing(self):
        '''Размножение'''
        new_persons = []
        # случайным образом спариваем особей пока не восстановим популяцию
        while len(new_persons + self.population) < self.population_count:
            new_person = choice(self.population) * choice(self.population)
            if new_person.prod == self.target and new_person.properties[0] >= new_person.properties[-1]:
                # найдена лучшая особь. инициируем завершение работы
                self.best_person = new_person
                break
            new_persons.append(new_person)
        self.population += new_persons


@click.command()
@click.option('--target', default=907200, help='Произведение всех параметров должно быть близким к этому числу')
@click.option('--properties_count', default=8, help='Количество параметров')
@click.option('--population_count', default=0, help='Количество особей в популяции')
@click.option('--depth', default=0, help='Максимальное количество поколений (итераций отбора/скрещивания)')
def run(target, properties_count, population_count, depth):
    if not population_count or not depth:
        model = load_model('model.h5')
        prediction = model.predict(np.array([[target, properties_count]]))
        population_count = int(prediction[0][0])
        depth = int(prediction[0][1])
        print(population_count, depth)
    farm = Farm(
        target = target,
        properties_count = properties_count,
        population_count = population_count,
        depth = depth,
    )
    result = farm.run()
    for k, v in result.items():
        if k in['depth', 'difference']:
            continue
        print(f'{k}: {v}')


if __name__ == '__main__':
    run()
