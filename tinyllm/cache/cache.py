import hashlib

from tinyllm import get_logger
from tinyllm.util import os_util
from tinyllm.util.ai_util import get_embedding, top_n_similar

logger = get_logger(name='default')

class LocalFilesCache:
    def __init__(self,
                 source_dir,
                 cache_path):
        logger.info(f"LocalFilesCache: Initializing cache at {cache_path}")
        if os_util.fileExists(cache_path):
            self.cache = os_util.loadJson(cache_path)
            if source_dir not in self.cache.keys():
                self.cache[source_dir] = {}
        else:
            self.cache = {source_dir: {}}
        self.cache_path = cache_path
        self.source_dir = source_dir
        self.persist()

    def get(self, key):
        if key in self.cache:
            return self.cache[self.source_dir][key]

    def set(self, key, value):
        self.cache[self.source_dir][self.source_dir][key] = value

    def delete(self, key):
        if key in self.cache[self.source_dir]:
            del self.cache[self.source_dir][key]

    def clear(self):
        self.cache[self.source_dir].clear()
        self.persist()

    def get_file_hash(self, file_path):
        with open(file_path, 'rb') as file:
            file_content = file.read()
            file_hash = hashlib.md5(file_content).hexdigest()
        return file_hash

    def get_file_content(self, file_path):
        with open(file_path, 'r') as f:
            return f.read()

    def persist(self):
        os_util.saveJson(self.cache, self.cache_path)

    def generate_cache(self,
                       file_path):
        logger.info(f"LocalFilesCache: Generating cache for {file_path}")
        new_file_hash = self.get_file_hash(file_path)
        if file_path in self.cache[self.source_dir].keys():
            last_file_hash = self.cache[self.source_dir][file_path]['file_hash']
            if new_file_hash != last_file_hash:
                self.update_file_cache(file_path,
                                       new_file_hash)
        else:
            self.update_file_cache(file_path,
                                   new_file_hash)

    def update_file_cache(self,
                          file_path,
                          file_hash):
        file_content = self.get_file_content(file_path)
        if file_content != '':
            embedding = get_embedding(file_content)
            cache_content = {
                'file_hash': file_hash,
                'embedding': embedding
            }
            self.cache[self.source_dir][file_path] = cache_content

    def refresh_cache(self,
                      only_missing=True):
        relevant_files = os_util.listDir(self.source_dir, recursive=True, formats=['py','md'])
        relevant_files = [file for file in relevant_files if '__init__' not in file]
        for file_path in self.cache[self.source_dir].keys():
            if file_path not in relevant_files:
                del self.cache[self.source_dir][file_path]

        if only_missing:
            relevant_files = [file for file in relevant_files if file not in self.cache[self.source_dir].keys()]
        for file in relevant_files:
            self.generate_cache(file)
        self.persist()


    def get_similar_files(self,
                          message,
                          n=5):
        content_embedding = get_embedding(message)
        embeddings_list = [self.cache[self.source_dir][file_path]['embedding'] for file_path in self.cache[self.source_dir].keys()]
        similar_embeddings_indices = top_n_similar(content_embedding, embeddings_list, n)
        similar_files_paths = [list(self.cache[self.source_dir].keys())[i] for i in similar_embeddings_indices]
        similar_content = [self.get_file_content(file_path) for file_path in similar_files_paths]
        return similar_content