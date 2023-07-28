import gzip
import io


def gzip_compress(text):
    text_bytes = text.encode('utf-8')
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as f:
        f.write(text_bytes)
    compressed_bytes = buf.getvalue()
    return compressed_bytes


def normalized_compression_distance(hash1, hash2):
    c_hash1 = len(gzip_compress(hash1))
    c_hash2 = len(gzip_compress(hash2))
    combined = hash1 + hash2
    c_combined = len(gzip_compress(combined))
    ncd = (c_combined - min(c_hash1, c_hash2)) / max(c_hash1, c_hash2)
    return ncd


def top_n_similar(input_doc, doc_list, n=2):
    ncd_values = [(doc, normalized_compression_distance(input_doc, doc)) for doc in doc_list]
    ncd_values.sort(key=lambda x: x[1])
    top_five = ncd_values[:n]
    return [doc[0] for doc in top_five]

