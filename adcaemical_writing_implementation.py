import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Load CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)

class Node:
    def __init__(self, node_type, params=None):
        self.node_type = node_type  # 'conv', 'pool', 'dense'
        self.params = params if params else {}

class Connection:
    def __init__(self, in_node, out_node):
        self.in_node = in_node
        self.out_node = out_node

class Genome:
    def __init__(self):
        self.nodes = []
        self.connections = []
        self.fitness = None

    def mutate(self, max_nodes=20):
        mutation_type = random.choice(['add_node', 'modify_node', 'add_connection'])
        if mutation_type == 'add_node' and len(self.nodes) < max_nodes:
            self.add_node()
        elif mutation_type == 'modify_node':
            self.modify_node()
        elif mutation_type == 'add_connection' and len(self.nodes) > 1:
            self.add_connection()

    def add_node(self):
        node_type = random.choice(['conv', 'pool', 'dense'])
        if node_type == 'conv':
            params = {
                'filters': random.choice([16, 32, 64]),
                'kernel_size': random.choice([3, 5]),
                'activation': random.choice(['relu', 'tanh'])
            }
        elif node_type == 'pool':
            params = {'pool_size': random.choice([2, 3])}
        elif node_type == 'dense':
            params = {'units': random.choice([64, 128, 256]), 'activation': 'relu'}

        new_node_id = len(self.nodes)
        new_node = Node(node_type, params)
        self.nodes.append(new_node)

        # Ensure the new node is connected
        if len(self.nodes) > 1:
            existing_node = random.choice(range(len(self.nodes) - 1))  # Exclude the new node
            self.connections.append(Connection(existing_node, new_node_id))

    def modify_node(self):
        if not self.nodes:
            return
        node = random.choice(self.nodes)
        if node.node_type == 'conv':
            node.params['filters'] = random.choice([16, 32, 64])
            node.params['activation'] = random.choice(['relu', 'tanh'])
        elif node.node_type == 'pool':
            node.params['pool_size'] = random.choice([2, 3])
        elif node.node_type == 'dense':
            node.params['units'] = random.choice([64, 128, 256])
            node.params['activation'] = random.choice(['relu', 'tanh'])

    def add_connection(self):
        if len(self.nodes) < 2:
            return
        in_node, out_node = random.sample(range(len(self.nodes)), 2)
        if in_node != out_node:  # Prevent self-loops
            self.connections.append(Connection(in_node, out_node))

def build_model(genome):
    model = models.Sequential()
    model.add(layers.Input(shape=(32, 32, 3)))

    has_flattened = False
    for node in genome.nodes:
        if node.node_type == 'conv' and not has_flattened:
            model.add(layers.Conv2D(
                filters=node.params['filters'],
                kernel_size=node.params['kernel_size'],
                activation=node.params['activation']
            ))
        elif node.node_type == 'pool' and not has_flattened:
            model.add(layers.MaxPooling2D(pool_size=node.params['pool_size']))
        elif node.node_type == 'dense':
            if not has_flattened:
                model.add(layers.Flatten())
                has_flattened = True
            model.add(layers.Dense(
                units=node.params['units'],
                activation=node.params['activation']
            ))

    model.add(layers.Dense(10, activation='softmax'))
    return model

def evaluate_fitness(genome):
    try:
        model = build_model(genome)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train[:5000], y_train[:5000], epochs=1, batch_size=32, verbose=0)
        _, accuracy = model.evaluate(x_test[:1000], y_test[:1000], verbose=0)
        genome.fitness = accuracy
    except Exception as e:
        print(f"Error evaluating genome: {e}")
        genome.fitness = 0.0

class Population:
    def __init__(self, size, max_nodes=20):
        self.genomes = [self.create_initial_genome() for _ in range(size)]
        self.generation = 0
        self.max_nodes = max_nodes

    def create_initial_genome(self):
        genome = Genome()
        genome.nodes.append(Node('conv', {'filters': 16, 'kernel_size': 3, 'activation': 'relu'}))
        genome.nodes.append(Node('pool', {'pool_size': 2}))
        genome.nodes.append(Node('dense', {'units': 128, 'activation': 'relu'}))
        genome.connections.append(Connection(0, 1))
        genome.connections.append(Connection(1, 2))
        return genome

    def evolve(self, elitism=2, mutation_prob=0.9):
        for genome in self.genomes:
            if genome.fitness is None:
                evaluate_fitness(genome)

        self.genomes.sort(key=lambda g: g.fitness if g.fitness is not None else 0.0, reverse=True)

        new_genomes = self.genomes[:elitism]
        for _ in range(len(self.genomes) - elitism):
            parent = random.choice(new_genomes)
            child = self.clone_genome(parent)
            if random.random() < mutation_prob:
                child.mutate(max_nodes=self.max_nodes)
            new_genomes.append(child)

        self.genomes = new_genomes
        self.generation += 1

    def clone_genome(self, genome):
        new_genome = Genome()
        new_genome.nodes = [Node(node.node_type, node.params.copy()) for node in genome.nodes]
        new_genome.connections = [Connection(conn.in_node, conn.out_node) for conn in genome.connections]
        return new_genome

def prune_disconnected_nodes(genome):
    """
    Remove nodes that are not part of any computational graph.
    Args:
        genome (Genome): The genome to prune.
    """
    G = nx.DiGraph()

    # Add nodes and edges to graph
    for i in range(len(genome.nodes)):
        G.add_node(i)
    for conn in genome.connections:
        G.add_edge(conn.in_node, conn.out_node)

    # Find reachable nodes from the first node (input node)
    reachable = set(nx.descendants(G, 0)) | {0}

    # Remove unreachable nodes
    genome.nodes = [node for i, node in enumerate(genome.nodes) if i in reachable]
    genome.connections = [conn for conn in genome.connections if conn.in_node in reachable and conn.out_node in reachable]

def tree_layout(graph):
    """
    Arrange nodes in a tree-like structure based on parent-child relationships.
    """
    from collections import defaultdict

    levels = defaultdict(list)
    for node in graph.nodes:
        level = len(nx.ancestors(graph, node))
        levels[level].append(node)

    pos = {}
    y_spacing = -1
    for y, (layer, nodes) in enumerate(sorted(levels.items())):
        x_spacing = 2.0
        for x, node in enumerate(nodes):
            pos[node] = (x * x_spacing, y * y_spacing)

    return pos

def visualize_genome(genome, generation, save_path=None):
    G = nx.DiGraph()

    for i, node in enumerate(genome.nodes):
        label = f"{node.node_type.upper()}\n{node.params}"
        G.add_node(i, label=label)

    for conn in genome.connections:
        if conn.in_node < len(genome.nodes) and conn.out_node < len(genome.nodes):
            G.add_edge(conn.in_node, conn.out_node)

    labels = nx.get_node_attributes(G, 'label')
    pos = tree_layout(G)

    plt.figure(figsize=(14, 10))
    nx.draw(
        G, pos, with_labels=True, labels=labels,
        node_size=2000, node_color='lightblue', font_size=8, font_weight='bold',
        arrowsize=10
    )
    plt.title(f"Best Genome Architecture - Generation {generation}", fontsize=16)

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
    else:
        plt.show()

# Main Evolution Loop
population = Population(size=30, max_nodes=25)
for generation in range(100):
    population.evolve(elitism=2, mutation_prob=0.9)
    best_genome = population.genomes[0]
    best_fitness = best_genome.fitness if best_genome.fitness is not None else 0.0
    print(f"Generation {population.generation}, Best Fitness: {best_fitness}")

    # Prune disconnected nodes before visualization
    prune_disconnected_nodes(best_genome)

# Visualize the best genome
visualize_genome(best_genome, generation=population.generation, save_path='best_genome/figure')
