import networkx as nx
import matplotlib.pyplot as plt

# Завдання 1

# Створимо порожній граф
G = nx.Graph()

# Додамо міста як вершини графа
cities = ["Відень", "Грац", "Зальцбург", "Лінц", "Інсбрук", "Клагенфурт", "Брегенц",
          "Маріацелль", "Хайлігенблют", "Гальштат"]
G.add_nodes_from(cities)

# Додамо дороги (ребра) з вагами між містами (хвилини в дорозі)
roads = [("Відень", "Грац", 130), ("Відень", "Лінц", 120), ("Відень", "Маріацелль", 125),
         ("Маріацелль", "Хайлігенблют", 260), ("Маріацелль", "Гальштат", 160), ("Маріацелль", "Лінц", 110),
         ("Маріацелль", "Грац", 85), ("Грац", "Клагенфурт", 90), ("Грац", "Лінц", 140),
         ("Грац", "Гальштат", 130), ("Клагенфурт", "Хайлігенблют", 115), ("Клагенфурт", "Гальштат", 150),
         ("Клагенфурт", "Лінц", 190), ("Клагенфурт", "Зальцбург", 150), ("Лінц", "Зальцбург", 85),
         ("Лінц", "Гальштат", 90), ("Зальцбург", "Гальштат", 75), ("Зальцбург", "Інсбрук", 130),
         ("Зальцбург", "Хайлігенблют", 140), ("Брегенц", "Інсбрук", 145), ("Хайлігенблют", "Інсбрук", 190),
         ("Хайлігенблют", "Гальштат", 155)]

G.add_weighted_edges_from(roads)

# Візуалізуємо граф
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=2000, font_size=12, font_weight='bold')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Транспортна мережа Австрії")
plt.show()

# Аналіз основних характеристик графа
print("Кількість міст (вершин) у мережі:", G.number_of_nodes())
print("Кількість доріг (ребер) у мережі:", G.number_of_edges())
print("Ступінь вершин (середня кількість зв'язків на місто):", sum(dict(G.degree()).values()) / G.number_of_nodes())

# Додатковий аналіз
# Середня довжина шляху
average_path_length = nx.average_shortest_path_length(G, weight='weight')
print("Середня довжина шляху:", average_path_length, "хв")

# Діаметр графа = максимальну відстань, щоб дістатися від однієї вершини до іншої в найгіршому випадку
diameter = nx.diameter(G)
print("Діаметр графа:", diameter)

# Коефіцієнт кластеризації
clustering_coefficients = nx.clustering(G)
average_clustering_coefficient = nx.average_clustering(G)
print("Коефіцієнти кластеризації вершин:", clustering_coefficients)
print("Середній коефіцієнт кластеризації:", average_clustering_coefficient)

# Центральність
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
closeness_centrality = nx.closeness_centrality(G, distance='weight')
print("Degree Centrality:", degree_centrality)
print("Betweenness Centrality:", betweenness_centrality)
print("Closeness Centrality:", closeness_centrality)

# Компоненти зв'язності
connected_components = list(nx.connected_components(G))
print("Кількість компонент зв'язності:", len(connected_components))
print("Розмір кожної компоненти зв'язності:", [len(component) for component in connected_components])

# Завдання 2

# DFS (Depth-First Search) алгоритм
def dfs(graph, start, end, path=[], visited=set()):
    path = path + [start]
    visited.add(start)
    if start == end:
        return path
    for neighbor in graph.neighbors(start):
        if neighbor not in visited:
            new_path = dfs(graph, neighbor, end, path, visited)
            if new_path:
                return new_path
    return None

# BFS (Breadth-First Search) алгоритм
def bfs(graph, start, end):
    queue = [[start]]
    visited = set()
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == end:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph.neighbors(node):
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    return None

# Знаходження шляхів за допомогою DFS та BFS
start_node = "Відень"
end_node = "Брегенц"

dfs_path = dfs(G, start_node, end_node)
bfs_path = bfs(G, start_node, end_node)

print("Шлях, знайдений за допомогою DFS:", dfs_path)
print("Шлях, знайдений за допомогою BFS:", bfs_path)


# Завдання 3

# Реалізація алгоритму Дейкстри
def dijkstra(graph, start):
    # Ініціалізація
    shortest_paths = {start: (None, 0)}
    current_node = start
    visited = set()
    
    while current_node is not None:
        visited.add(current_node)
        destinations = graph.neighbors(current_node)
        weight_to_current_node = shortest_paths[current_node][1]
        
        for next_node in destinations:
            weight = graph[current_node][next_node]['weight'] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return shortest_paths
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    return shortest_paths

# Знаходження найкоротших шляхів між всіма вершинами
all_pairs_shortest_paths = {}
for city in cities:
    all_pairs_shortest_paths[city] = dijkstra(G, city)

# Виведення найкоротших шляхів між всіма парами вершин
for start in all_pairs_shortest_paths:
    print(f"\nНайкоротші шляхи з міста {start}:")
    for destination in all_pairs_shortest_paths[start]:
        if start != destination:
            path, weight = [], destination
            while weight is not None:
                path.append(weight)
                weight = all_pairs_shortest_paths[start][weight][0]
            path = path[::-1][1:]
            print(f"  до {destination}: {' -> '.join([start] + path)} (вага: {all_pairs_shortest_paths[start][destination][1]} хв)")
