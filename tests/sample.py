from py2neo import Graph, Node
import json

host = "483f5352.databases.neo4j.io",
port = 7687
user = "neo4j"
password = "3iU4hFam3Fe7CXMLxbeniSKi3Uh4B8AWhsXH4oJpW0E"
Graph(f"neo4j+s://{host}:{port}", auth=(user, password))


class Function:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def push_to_cypher(self):
        # Convert object to dict
        attributes_dict = vars(self)
        node = Node(self.__class__.__name__, **attributes_dict)
        graph.create(node)


class Primitive1(Function):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Primitive2(Function):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# Now you can create instances of your classes and push them to the graph:
p1 = Primitive1(attr1='value1', attr2='value2')
p1.push_to_cypher()

p2 = Primitive2(attr3='value3', attr4='value4')
p2.push_to_cypher()
