"""Array-backed running top-k for large sparse candidate searches."""
import numpy as np


def encode_target_id(identifier):
    prefix, separator, value = identifier.partition('-')
    if separator != '-' or prefix not in ('S2', 'S3') or not value.isdecimal():
        raise ValueError(f'Unexpected target ID: {identifier!r}')
    number = int(value)
    if number >= (1 << 62) - 1:
        raise ValueError('Target ID exceeds compact representation')
    return (number + 1) * 2 + (prefix == 'S3')


def decode_target_id(encoded):
    value = int(encoded)
    if value == 0:
        raise ValueError('Empty target slot')
    return f'S{3 if value & 1 else 2}-{(value >> 1) - 1}'


class CompactTopK:
    def __init__(self, rows, k):
        if rows < 1 or k < 1:
            raise ValueError('rows and k must be positive')
        self.k = k
        self.scores = np.full((rows, k), -np.inf, dtype=np.float32)
        self.ids = np.zeros((rows, k), dtype=np.uint64)

    def update(self, start, scores, target_ids):
        """Merge a CSR top-k chunk for consecutive reference rows."""
        rows = scores.shape[0]
        end = start + rows
        if start < 0 or end > len(self.scores):
            raise ValueError('Reference row range out of bounds')
        if scores.shape[1] != len(target_ids):
            raise ValueError('Target IDs do not match score columns')
        if rows == 0:
            return
        counts = np.diff(scores.indptr)
        if np.any(counts > self.k):
            raise ValueError('Chunk contains more than k scores per row')
        local_scores = np.full((rows, self.k), -np.inf, dtype=np.float32)
        local_ids = np.zeros((rows, self.k), dtype=np.uint64)
        positions = np.arange(scores.nnz, dtype=np.int64) - np.repeat(scores.indptr[:-1], counts)
        row_indexes = np.repeat(np.arange(rows), counts)
        local_scores[row_indexes, positions] = scores.data
        local_ids[row_indexes, positions] = target_ids[scores.indices]
        merged_scores = np.concatenate((self.scores[start:end], local_scores), axis=1)
        merged_ids = np.concatenate((self.ids[start:end], local_ids), axis=1)
        top = np.argpartition(merged_scores, -self.k, axis=1)[:, -self.k:]
        self.scores[start:end] = np.take_along_axis(merged_scores, top, axis=1)
        self.ids[start:end] = np.take_along_axis(merged_ids, top, axis=1)

    def ranked(self, row):
        order = np.lexsort((self.ids[row], self.scores[row]))[::-1]
        return [(float(self.scores[row, col]), decode_target_id(self.ids[row, col]))
                for col in order if self.ids[row, col] != 0]
