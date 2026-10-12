from flask import Flask, render_template, request, jsonify
from src.domain.models import DataObject
from src.services.clustering import ClusteringSystem

app = Flask(__name__)

# Множество из 12 объектов по 3 признакам
# Признаки: [Частота ЦП (ГГц), ОЗУ (ГБ), Вес (кг)]
dataset = [
    DataObject("1", [2.4, 8, 1.8], "Laptop A (Офисный)"),
    DataObject("2", [2.1, 4, 2.1], "Laptop B (Бюджетный)"),
    DataObject("3", [2.5, 8, 1.7], "Laptop C (Офисный)"),
    DataObject("4", [4.2, 32, 3.2], "Laptop D (Игровой)"),
    DataObject("5", [4.0, 16, 2.9], "Laptop E (Игровой)"),
    DataObject("6", [1.2, 4, 1.1], "Laptop F (Ультрабук)"),
    DataObject("7", [4.5, 32, 3.5], "Laptop G (Игровой)"),
    DataObject("8", [1.1, 8, 1.0], "Laptop H (Ультрабук)"),
    DataObject("9", [2.3, 8, 1.9], "Laptop I (Офисный)"),
    DataObject("10", [1.4, 8, 1.2], "Laptop J (Ультрабук)"),
    DataObject("11", [3.8, 16, 2.8], "Laptop K (Игровой)"),
    DataObject("12", [2.6, 16, 2.0], "Laptop L (Офисный)")
]

clustering_system = ClusteringSystem(dataset)


@app.route("/")
def index():
    return render_template("index.html")


def _prepare_response(clusters, prototypes):
    """Форматирует данные для отправки в веб-интерфейс."""
    result = []
    for c_id, objects in clusters.items():
        result.append({
            "cluster_id": c_id + 1,
            "prototype": [round(p, 2) for p in prototypes[c_id]],
            "objects": [{"name": obj.name, "features": obj.features} for obj in objects]
        })
    return result


@app.route("/api/cluster", methods=["POST"])
def cluster():
    data = request.json
    method = data.get("method", "kmeans")

    if method == "kmeans":
        k = int(data.get("k", 3))
        clusters, prototypes = clustering_system.kmeans(k)
    elif method == "maximin":
        clusters, prototypes = clustering_system.maximin()
    else:
        return jsonify({"error": "Unknown method"}), 400

    return jsonify({
        "clusters": _prepare_response(clusters, prototypes)
    })