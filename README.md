<p align="center">
    <h1 align="center">Оптимизатор mintaero</h1>
    </p>

<h4>Реализованная функциональность</h4>
<ul>
    <li>Решение задачи оптимизации расстановки самолетов по местам стоянок</li>
</ul> 
<h4>Особенность проекта в следующем:</h4>
<ul>
 <li>Возможность поиск оптимального решения. - "Найди самое лучшее решение"</li>
 <li>Возможность быстрого поиск менее оптимального решения.- "Найди хорошее решение, но быстро"</li>  
 <li>Возможность ограничения времени на поиск решения. - "Найди любое решение, но за X минут"</li>

 </ul>
<h4>Основной стек технологий:</h4>
<ul>
    <li>Python</li>
	<li>Pandas, MIP</li>
  
 </ul>




СРЕДА ЗАПУСКА
------------
1) Необходим установленый Python. Библиотеки pandas, mip, dataclasses


УСТАНОВКА
------------

### База данных

Отсутствует

### Выполнение миграций

Отсутствует

### Установка зависимостей проекта

Отсутствует


### Входные параметры сервиса
<li>data-dir - директория с входными данными. По-умолчанию "../data"</li>
<li>output - имя файла с результатом работы сервиса. По-умолчанию "../data/result.csv"</li>


### сервис поиск оптимального решения

файл src/optimize.py 

запуск:
python optimize.py

### сервис поиск быстрого оптимального решения

<li> файл для кластеризации МС - src/clusterize.py </li>
<li> файл оптимизатор - src/optimize_fast.py </li>

запуск:
<li>python clusterize.py</li>
<li>python optimize_fast.py</li>


###РАЗРАБОТЧИКИ

<h4>Мещеряков Артем d.karvallio@gmail.com </h4>
<h4>Шульцева Ольга shultseva.olga@gmail.com </h4>
<h4>Миценко Вадим v.mitsenko98@yandex.ru </h4>
