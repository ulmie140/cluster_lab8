import math
from typing import List, Dict, Tuple
from src.domain.models import DataObject


class ClusteringSystem:
    """
    Главный класс бизнес-логики для кластерного анализа.
    Отвечает за группировку объектов (ноутбуков) "вслепую"
    с помощью алгоритмов К-средних и Максимина.
    """

    def __init__(self, data: List[DataObject]):
        # Сохраняем переданное множество объектов, которые нужно разбить на группы
        self.data = data

        # Автоматически определяем размерность пространства (количество признаков).
        self.feature_count = len(data[0].features)

    def _distance(self, f1: List[float], f2: List[float]) -> float:
        """Евклидово расстояние между двумя точками."""
        return math.sqrt(sum((f1[i] - f2[i]) ** 2 for i in range(self.feature_count)))

    def kmeans(self, k: int) -> Tuple[Dict[int, List[DataObject]], List[List[float]]]:
        """
        Алгоритм К-средних.
        Разбивает объекты на заданное число кластеров K.
        """
        if not self.data or k <= 0 or k > len(self.data):
            return {}, []

        # Шаг 2: В качестве первоначальных прототипов берем первые K объектов
        prototypes = [list(obj.features) for obj in self.data[:k]]
        clusters = {i: [] for i in range(k)}

        while True:
            # Шаг 4: Отнесение каждого объекта к ближайшему кластеру
            new_clusters = {i: [] for i in range(k)}
            for obj in self.data:
                min_dist = float('inf')
                nearest_p = -1
                for i, p in enumerate(prototypes):
                    d = self._distance(obj.features, p)
                    if d < min_dist:
                        min_dist = d
                        nearest_p = i
                new_clusters[nearest_p].append(obj)

            # Шаг 5: Определение новых прототипов (среднее арифметическое)
            new_prototypes = []
            for i in range(k):
                cluster_objects = new_clusters[i]
                if not cluster_objects:
                    new_prototypes.append(prototypes[i])  # Если кластер пуст, оставляем старый прототип
                    continue

                avg_features = [0.0] * self.feature_count
                for obj in cluster_objects:
                    for j in range(self.feature_count):
                        avg_features[j] += obj.features[j]

                avg_features = [val / len(cluster_objects) for val in avg_features]
                new_prototypes.append(avg_features)

            # Шаг 6: Проверка условия завершения (прототипы не изменились)
            changed = False
            for i in range(k):
                for j in range(self.feature_count):
                    if abs(prototypes[i][j] - new_prototypes[i][j]) > 1e-6:
                        changed = True
                        break

            if not changed:
                clusters = new_clusters
                break

            prototypes = new_prototypes
            clusters = new_clusters

        return clusters, prototypes

    def maximin(self) -> Tuple[Dict[int, List[DataObject]], List[List[float]]]:
        """
        Алгоритм максимина.
        Количество кластеров заранее неизвестно и определяется автоматически.
        """
        if not self.data: return {}, []

        # Шаг 1: Первый объект становится прототипом первого кластера
        prototypes = [list(self.data[0].features)]

        # Шаг 2: Находим объект, наиболее удаленный от первого прототипа
        max_d = -1
        p2_features = None
        for obj in self.data:
            d = self._distance(prototypes[0], obj.features)
            if d > max_d:
                max_d = d
                p2_features = list(obj.features)
        prototypes.append(p2_features)

        # Шаг 3: Пороговое расстояние (половина расстояния между P1 и P2)
        threshold = self._distance(prototypes[0], prototypes[1]) / 2.0

        while True:
            # Шаг 4: Отнесение каждого объекта к ближайшему прототипу
            clusters = {i: [] for i in range(len(prototypes))}
            for obj in self.data:
                min_d = float('inf')
                nearest_p = -1
                for i, p in enumerate(prototypes):
                    d = self._distance(obj.features, p)
                    if d < min_d:
                        min_d = d
                        nearest_p = i
                clusters[nearest_p].append(obj)

            # Шаг 5: Поиск наиболее удаленного объекта в каждом кластере
            new_prototypes = []
            for i, p in enumerate(prototypes):
                max_d = -1
                furthest_obj = None
                for obj in clusters[i]:
                    d = self._distance(p, obj.features)
                    if d > max_d:
                        max_d = d
                        furthest_obj = obj

                # Шаг 6: Проверка условия образования новых кластеров
                if furthest_obj is not None and max_d >= threshold:
                    new_prototypes.append(list(furthest_obj.features))

            # Если новых кластеров нет, алгоритм завершается
            if not new_prototypes:
                break

            prototypes.extend(new_prototypes)

            # Шаг 7: Новое пороговое расстояние (половина среднего арифметического расстояний между всеми прототипами)
            sum_d = 0.0
            count = 0
            k = len(prototypes)
            for i in range(k - 1):
                for j in range(i + 1, k):
                    sum_d += self._distance(prototypes[i], prototypes[j])
                    count += 1
            threshold = (sum_d / count) / 2.0

        return clusters, prototypes