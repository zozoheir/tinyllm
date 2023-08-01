import os
import pathspec
from sqlalchemy import text
from sqlalchemy.orm import Session

from tinyllm.gitignore import gitignore_content

from tinyllm.util import os_util


class LocalDirCache:
    def __init__(self, directory_name, vector_collection, collection_name='tinyllm'):
        self.cache = {}
        self.directory_name = directory_name
        self.collection_name = collection_name
        self.vector_store = vector_collection
        self.connection = vector_collection.connect()
        self.load_from_dir()

    def load_from_dir(self):
        spec = pathspec.PathSpec.from_lines('gitwildmatch', gitignore_content.splitlines())
        file_paths = os_util.listDir(self.directory_name, recursive=True, formats=['md','py'])
        filtered_files = [file_path for file_path in file_paths if not spec.match_file(file_path)]
        for file_path in filtered_files:
            content = self.get_file_content(file_path)
            if content:
                self.cache[file_path] = content
        print(f"Loaded {len(filtered_files)} files from directory")

    def get_file_content(self, file_path):
        with open(file_path, 'r') as file:
            content = file.read().strip()
            return content

    def embedded_files(self):
        with Session(self.connection) as session:
            results = session.execute(text("select embedding_metadata from embeddings")).fetchall()
            return [metadata['file_path'] for result in results for metadata in result if
                    'file_path' in metadata.keys()]

    def add_missing_files(self):
        embedded_files = self.embedded_files()
        to_embed = [i for i in self.cache.keys() if i not in embedded_files]
        self.vector_store.add_texts(
            texts=[self.cache[file_path] for file_path in to_embed],
            metadatas=[{'file_path': file_path} for file_path in to_embed],
        )
        print("Added missing files")

    def delete_old_files_from_db(self):
        embedded_files = self.embedded_files()
        files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.directory_name) for f in filenames]
        to_delete = [i for i in embedded_files if i not in files]
        to_delete_sql = ', '.join([f"'{item}'" for item in to_delete])
        if len(to_delete_sql) == 0:
            return

        delete_query = f"""
        DELETE FROM public.langchain_pg_embedding
        WHERE cmetadata->>'file_path' = ANY(ARRAY[{to_delete_sql}]);
        """
        with Session(self.connection) as session:
            session.execute(text(delete_query))
            session.commit()
        print("Deleted old files from DB")

    def refresh_cache(self):
        self.load_from_dir()
        self.add_missing_files()
        self.delete_old_files_from_db()
        print("Cache refreshed")
