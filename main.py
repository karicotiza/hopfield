import numpy as np  # Импорт модуля numpy для работы с математическими функциями и матрицами


class Hopfield:  # Создание класса Hopfield для работы с сетью Хопфилда
    """
    Hopfield(images, test_image, view="short")

        Создаёт сеть Хопфилда с непрерывным состоянием, дискретным временем
        и синхронным режимом. Сразу же производит все вычисления. В экземпляр
        передавать обычные одномерные массивы заданные через квадратные скобки

        Пример
        ----------
        net = Hopfield(images=[[-1, 1, -1, 1],
                [1, -1, 1, 1],
                [-1, 1, -1, -1]],
            test_image=[1, -1, 1, -1])

        Параметры
        ----------
        images - эталонные образы
        test_image - образ для проверки
        view - максимальный размер выводимых в консоль элементов

        Функции
        ----------
        print_x_vectors() - Вывести векторы "x" (эталонные образы), (Шаг 1)
        print_xt_multiply_x() - Вывести векторы "x" умноженные на транспонированные векторы "x" (Шаг 2)
        print_w() - Вывести сумму всех векторов "x" умноженных на транспонированные векторы "x"
                    (Матрица весовых коэффициентов), (Матрица "W"), (Шаг 3)
        print_zeroed_w() - Вывести матрицу "W" с нулями на главной диагонали (зануленную) (Шаг 4)
        print_y_vector() - Вывести вектор "y" (образ для проверки) (Шаг 5)
        print_w_multiply_y() - Вывести произведение матрицы "W" на вектор "y" (Шаг 6)
        print_tanh_w_multiply_y() - Вывести результат применения функции активации
                                    к произведению матрицы "W" на вектор "y" (Шаг 7)
        print_result() - Вывести результат (распознанный образ и количество затраченных итераций)
    """

    def __init__(self, images, test_image, view="short"):  # Функция инициализации экземпляра объекта класса

        # Техническая часть
        self.print_option = view  # Инициализация переменной view для настройки вывода результатов в консоль
        self.set_print_options()  # Вызов функции для настройки вывода результатов в консоль

        # логическая часть
        self.x_vectors = []  # Инициализация массива хранящего векторы "x" (эталонные образы)
        self.xt_multiply_x = []  # Инициализация массива хранящего значения произведения матр. "X" транс. на матр. "X"
        self.w = []  # Инициализация массива хранящего матрицу "W" (матрицу весовых коэффициентов)
        self.zeroed_w = []  # Инициализация массива хранящего зануленную матрицу "W" (матрицу весовых коэффициентов)
        self.y_vector = []  # Инициализация массива хранящего вектор "y" (тестовый образ)
        self.w_multiply_y = []  # Инициализация массива хранящего произведение зануленной матрицы "W" на вектор "y"
        self.tanh_w_multiply_y = []  # Ин. массива хранящего значения ф-и "tanh" от пр. зан. матр. "W" на век. "y"
        self.recognized_image = 0  # Инициализация номера распознанного образа

        self.create_x_vectors(images)  # Вызов функции обработки векторов "x"
        self.calculate_xt_multiply_x()  # Вызов функции умножения матр. "x" транс. на матр. "x"
        self.calculate_w()  # Вызов функции расчёта матрицы "W"
        self.zero_out_x()  # Вызов функции зануления матрицы "W"
        self.create_y_vector(test_image)  # Вызов функции обработки вектора "y"
        self.calculate_zeroed_w_multiply_y()  # Вызов функции вычисления произв. зануленной матрицы "W" на вектор "y"
        self.recognize()  # Вызов функции определения номера распознанного образа

    # Функции отвечающие за настройку вывода

    def set_print_options(self):  # Функция для настройки вывода результатов в консоль
        if self.print_option == "full":  # Если при переменная "print_option" равна "full"
            np.set_printoptions(threshold=np.inf)  # Выводить в консоль большие матрицы полностью

    # Функции отвечающие за алгоритм

    def create_x_vectors(self, vectors):  # Функция обработки векторов "x"
        for vector in vectors:  # Для каждого вектора переданного в экземпляр
            numpy_array = np.array(vector)  # Преобразовать вектор в массив numpy
            self.x_vectors.append(numpy_array)  # Добавить массив numpy в массив хранящий массивы "x"

    def calculate_xt_multiply_x(self):  # Функция умножения матрицы X транспонированной на матрицу X
        for numpy_array in self.x_vectors:  # Для каждого вектора в массиве векторов "x"
            xt = numpy_array.reshape(numpy_array.size, 1)  # Создание транспонированной матрицы "xt" из вектора "x"
            xt_multiply_x = xt * numpy_array  # Умножение матрицы "xt" на матрицу "x"
            self.xt_multiply_x.append(xt_multiply_x)  # Добавление результата в массив

    def calculate_w(self):  # Функция расчёта матрицы W
        w = 0  # Создание временной переменной для суммирования матриц
        for matrix in self.xt_multiply_x:  # Для каждой матрицы из массива матриц "xt * t"
            w = w + matrix  # Сумма матриц равна результату сложения существующей матрицы "W" и текущей матрицы "matrix"
        self.w = w  # Присвоение переменной w значения суммы матриц

    def zero_out_x(self):  # Функция зануления матрицы W
        self.zeroed_w = self.w * (  # Матрица W поэлементно умножается на единичную матрицу, где на гл. диагонали нули
                np.ones(self.x_vectors[0].size, int) - np.identity(self.x_vectors[0].size, int)
        )

    def create_y_vector(self, vector):  # Функции обработки вектора "y"
        numpy_array = np.array(vector)  # Преобразование вектора "y" в массив "numpy"
        self.y_vector.append(numpy_array)  # Добавить массив numpy в массив хранящий массивы "y"

    def calculate_zeroed_w_multiply_y(self):  # Функция вычисления произведения зануленной матрицы "W" на вектор "y"
        for numpy_array in self.y_vector:  # Для каждого вектора "y" (тестового образа)
            y = numpy_array.reshape(numpy_array.size, 1)  # Формируем матрицу размера (n, 1) из вектора размера (n)
            zeroed_w_multiply_y = np.matmul(self.zeroed_w, y)  # Умножаем матрицу "W" на матрицу "y"
            self.w_multiply_y.append(zeroed_w_multiply_y)  # Добавляем результат умножения в массив
            tanh_zeroed_w_multiply_y = np.sign(zeroed_w_multiply_y)  # Находим "tanh" для эл. в полученной матрице
            self.tanh_w_multiply_y.append(tanh_zeroed_w_multiply_y)  # Добавляем результат вычислений tanh в массив
            temporary_variable = np.zeros(1)  # Инициализируем временную переменную для сравнения матриц

            while True:  # Бесконечный цикл
                zeroed_w_multiply_y = np.matmul(self.zeroed_w, tanh_zeroed_w_multiply_y)  # Умножаем W на tanh(y)
                tanh_zeroed_w_multiply_y = np.sign(zeroed_w_multiply_y)  # Обновляем матрицу tanh(y)
                # TODO: Я тут np.tanh(zeroed_w_multiply_y) поменял на
                #       np.sign(1 + np.sign(zeroed_w_multiply_y)) в двух местах,
                #       надо вернуть как было

                # Функция сравнения состояния матрицы в момент "t" и "t-1"
                requirement = (  # Если первые четыре символа после запятой совпадают
                        np.around(temporary_variable, 4) == np.around(tanh_zeroed_w_multiply_y, 4)
                )  # то считается, что сеть достигла релаксации
                counter = 0
                for element in requirement:
                    if element == [False]:
                        counter = counter + 1

                if counter == 0:  # Если матрица tanh(y) на шаге n совпадает с n - 1
                    break  # Завершить цикл

                temporary_variable = tanh_zeroed_w_multiply_y  # Обновить временную переменную
                self.w_multiply_y.append(zeroed_w_multiply_y)  # Добавить матрицу W в список
                self.tanh_w_multiply_y.append(tanh_zeroed_w_multiply_y)  # Добавить tanh(y) в список
                print(tanh_zeroed_w_multiply_y)

    def recognize(self):  # Функция определения найденного образа
        recognized_image = np.sign(self.tanh_w_multiply_y[-1])  # Инициализация распознанного образа
        recognized_image = recognized_image.reshape(1, recognized_image.size)[0]  # Подготовка вектора
        recognized_image = recognized_image.astype(int)  # Подготовка вектора
        count = 1  # Инициализация вспомогательной переменной

        for numpy_array in self.x_vectors:  # Для каждого вектора "x"
            numpy_array = np.sign(numpy_array)  # Подготовка вектора "x"
            numpy_array = np.around(numpy_array, 1)  # Подготовка вектора "x"
            numpy_array = numpy_array.astype(int)  # Подготовка вектора "x"

            temporary_counter_1 = 0  # Инициализация вспомогательной переменной
            temporary_counter_2 = 0  # Инициализация вспомогательной переменной
            negative_numpy_array = numpy_array * -1

            for i in range(recognized_image.size):  # Для каждого элемента в распознанном образе
                # Если все элементы распознанного образа и вектора "x" № "i" равны
                if recognized_image[i] == numpy_array[i]:
                    temporary_counter_1 = temporary_counter_1 + 1  # Увеличить временную переменную на 1
                    if temporary_counter_1 >= recognized_image.size - 1:  # Если все элементы равны
                        self.recognized_image = count  # Вернуть переменную
                        break  # Прекратить поиск

                # Если все элементы распознанного обр. и вектора "x" № "i" умноженного на (-1)равны
                if recognized_image[i] == negative_numpy_array[i]:
                    temporary_counter_2 = temporary_counter_2 + 1  # Увеличить временную переменную на 1
                    if temporary_counter_2 >= recognized_image.size - 1:  # Если все элементы равны
                        self.recognized_image = - count  # Вернуть переменную
                        break  # Прекратить поиск

            count = count + 1

    # Функции отвечающие за вывод данных в консоль

    def print_x_vectors(self):  # Вывести в консоль векторы "x"
        for numpy_array in self.x_vectors:  # Для каждого массива numpy в массиве содержащем векторы "x"
            print(numpy_array)  # Вывести в консоль массив numpy

    def print_xt_multiply_x(self):  # Вывести в консоль значения произведения матрицы "xt" и "x"
        for numpy_array in self.xt_multiply_x:  # Для каждого массива numpy в массиве содержащем произведения "xt" и "x"
            print(numpy_array)  # Вывести в консоль массив numpy

    def print_w(self):  # Вывести в консоль матрицу "W"
        print(self.w)  # Вывести в консоль матрицу "W"

    def print_zeroed_w(self):  # Вывести в консоль зануленную матрицу "W"
        print(self.zeroed_w)  # Вывести в консоль зануленную матрицу "W"

    def print_y_vector(self):  # Вывести в консоль вектор "y"
        print(self.y_vector[0])  # Вывести в консоль вектор "y"

    def print_w_multiply_y(self):  # Вывести в консоль значения произведений матрицы "W" на вектор "y"
        for numpy_array in self.w_multiply_y:  # Для каждого массива numpy в массиве содержащем результаты "W * y"
            print(numpy_array)  # Вывести в консоль массив numpy

    def print_tanh_w_multiply_y(self):  # Вывести в консоль значения произведений зануленной матрицы "W" на вектор "y"
        for numpy_array in self.tanh_w_multiply_y:  # Для каждого массива numpy в массиве содержащем "tanh(W * y)"
            print(numpy_array)  # Вывести в консоль массив numpy

    def print_result(self):  # Вывести в консоль результат
        print(f"Итераций выполнено: {len(self.tanh_w_multiply_y) + 1}")  # Итерации понадобившиеся для релаксации сети
        print(f"Последняя итерация tanh(W * y): ")
        for element in self.tanh_w_multiply_y[-1]:  # Последний результат "tanh(W * y)"
            print(" ", element)
        if self.recognized_image > 0:
            print(f"Тестовый образ распознан как образ №{self.recognized_image}")
        else:
            print(f"Тестовый образ распознан как негатив образа №{self.recognized_image * -1}")


def main():  # Основная функция
    net = Hopfield(  # Создание экземпляра "net" для проверки
        images=[[  # Образы
            -0.5, 0.5, -0.5,  # Символ "A"
            0.5, 0.5, 0.5,
            0.5, -0.5, 0.5
        ], [
            0.5, 0.5, 0.5,  # Символ "C"
            0.5, -0.5, -0.5,
            0.5, 0.5, 0.5,
        ], [
            0.5, 0.5, -0.5,  # Символ "D"
            0.5, -0.5, 0.5,
            0.5, 0.5, -0.5
        ]],
        test_image=[  # Негатив символа "C"
            -0.8, -0.8, -0.8,
            0.2, 0.2, 0.2,
            0.2, -0.8, -0.8
        ]
    )

    net.print_zeroed_w()
    net.print_xt_multiply_x()
    net.print_w()
    net.print_zeroed_w()
    net.print_tanh_w_multiply_y()
    net.print_result()


if __name__ == "__main__":  # Если файл называется main.py
    main()  # Вызов главной функции
