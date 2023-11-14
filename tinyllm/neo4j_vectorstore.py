import uuid
from typing import List, Dict, Callable, Any, Iterable, Optional


class Neo4JVectorStore:
    COSINE_DISTANCE_STRATEGY = "COSINE"

    def __init__(
            self,
            *,
            embedding_function: Callable[[str], List[float]],
            url: str,
            username: str,
            password: str,
            database: str = "neo4j",
            index_name: str = "vectorstore",
            node_label: str = "Node",
            embedding_property_name: str = "embedding"
    ) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "Could not import neo4j package. "
                "Please install it with `pip install neo4j`."
            )

        self._driver = GraphDatabase.driver(url, auth=(username, password))
        self._database = database
        self.embedding_function = embedding_function
        self.index_name = index_name
        self.node_label = node_label
        self.embedding_property_name = embedding_property_name
        self._verify_connection()

    def _verify_connection(self):
        try:
            self._driver.verify_connectivity()
        except Exception as e:
            raise ConnectionError("Failed to connect to the Neo4j database") from e

    def create_new_index(self):
        with self._driver.session(database=self._database) as session:
            session.run(
                "CALL db.index.fulltext.createNodeIndex($index_name, [$node_label], "
                "[$embedding_property_name])",
                index_name=self.index_name,
                node_label=self.node_label,
                embedding_property_name=self.embedding_property_name
            )

    def retrieve_existing_index(self) -> bool:
        with self._driver.session(database=self._database) as session:
            result = session.run(
                "CALL db.index.fulltext.listAvailableAnalyzers"
            )
            indexes = result.data()
            for index in indexes:
                if index['name'] == self.index_name:
                    return True
            return False

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None
    ) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        embeddings = [self.embedding_function(text) for text in texts]

        query = (
            "UNWIND $data AS row "
            f"MERGE (n:{self.node_label} {{id: row.id}}) "
            f"SET n.{self.embedding_property_name} = row.embedding, "
            "n.metadata = row.metadata "
        )

        with self._driver.session(database=self._database) as session:
            session.run(
                query,
                data=[{
                    'id': id,
                    'embedding': embedding,
                    'metadata': metadata
                } for id, embedding, metadata in zip(ids, embeddings, metadatas)]
            )

        return ids

    def similarity_search(
            self,
            query: str,
            k: int = 4,
            index_filters: Optional[Dict[str, Dict[str, List[Any]]]] = None
    ) -> List[Dict[str, Any]]:
        embedding = self.embedding_function(query)
        match_clause = self._construct_match_clause(index_filters)

        search_query = (
            f"CALL db.index.fulltext.queryNodes($index_name, $query) "
            "YIELD node, score "
            f"{match_clause} "
            "RETURN node AS result, score "
            "ORDER BY score DESC "
            "LIMIT $k"
        )

        with self._driver.session(database=self._database) as session:
            results = session.run(
                search_query,
                index_name=self.index_name,
                query=embedding,
                k=k
            ).data()

        return results

    def _construct_match_clause(
            self,
            index_filters: Optional[Dict[str, Dict[str, List[Any]]]]
    ) -> str:
        if not index_filters:
            return ""

        clauses = []
        for node_name, filters in index_filters.items():
            for property_name, values in filters.items():
                filter_clause = (
                    f"WHERE n:{node_name} AND n.{property_name} IN $values "
                )
                clauses.append(filter_clause)

        return " ".join(clauses)

# Example usage
embedding_function = lambda x: [float(ord(c)) for c in x]  # Replace with real function
neo4j_store = Neo4JVectorStore(
    embedding_function=embedding_function,
    #url='neo4j+s://bab834f9.databases.neo4j.io:7687',
    #username='neo4j',
    #password='uyQBiLNbTWYZD3AkUSOxZkIgUnrnhbek4mJ3ex0q_cw'
)
neo4j_store.create_new_index()