import tkinter as tk
from tkinter import ttk

from genetic import Farm

from tensorflow.keras.models import load_model
import numpy as np


window = tk.Tk()
window.geometry('550x200')
window.title('Поиск множителей')
tab_control = ttk.Notebook(window)
model = load_model('model.h5')


def search():
    '''Найти множители произведения'''
    target = int(target_edit.get())
    properties_count = int(properties_count_edit.get())

    # предсказание недостающих параметров
    prediction = model.predict(np.array([[target, properties_count]]))
    population_count = int(prediction[0][0])
    depth = int(prediction[0][1])

    farm = Farm(
        target = target,
        properties_count = properties_count,
        population_count = population_count,
        depth = depth,
    )
    result = farm.run()
    result_edit.delete(0, tk.END)
    result_edit.insert(0, result['Найденный массив чисел'].replace(', ', ' * ') + f" = {result['Результат']}")


# Расстановка элементов на форме окна
# Вкладка клиенты
multipliers_search_tab = ttk.Frame(tab_control)
tab_control.add(multipliers_search_tab, text='Поиск множителей')
tab_control.pack(expand=1)

target_label = tk.Label(multipliers_search_tab, text='Произведение:')
target_edit = tk.Entry(multipliers_search_tab, width=30)
properties_count_label = tk.Label(multipliers_search_tab, text='Количество множителей:')
properties_count_edit = tk.Entry(multipliers_search_tab, width=30)
result_label = tk.Label(multipliers_search_tab, text='Найденный массив чисел:')
result_edit = tk.Entry(multipliers_search_tab, width=30)

add_client_button = tk.Button(
    multipliers_search_tab, 
    text='Найти', 
    command=search
)

target_label.grid(column=0, row=1)
target_edit.grid(column=1, row=1, columnspan=2)
properties_count_label.grid(column=0, row=2)
properties_count_edit.grid(column=1, row=2, columnspan=2)
result_label.grid(column=0, row=3)
result_edit.grid(column=1, row=3, columnspan=2)
add_client_button.grid(column=0, row=4)

window.mainloop()