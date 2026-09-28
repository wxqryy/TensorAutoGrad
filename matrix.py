def mul_on_vector(matrix, vector):
    if not matrix or not matrix[0]:
        return 0
    if len(matrix[0]) != len(vector):
        return 0
    res = []

    for row in matrix:
        if len(row) != len(vector):
            return 0
        s = 0
        for idx, e in enumerate(row):
            s += e * vector[idx]
        res.append(s)
    return res

def transpose(matrix):
    if not matrix or not matrix[0]:
        return 0
    result = [[0] * len(matrix) for _ in range(len(matrix[0]))]

    for row_id, row in enumerate(matrix):
        for column_id, value in enumerate(row):
            result[column_id][row_id] = value
    return result

def add(vec_a, vec_b):
    if len(vec_a) != len(vec_b):
        return
    return [vec_a[i] + vec_b[i] for i in range(len(vec_a))]

def sub(vec_a, vec_b):
    if len(vec_a) != len(vec_b):
        return 0
    return [vec_a[i] - vec_b[i] for i in range(len(vec_a))]

def mul_elementwise(vec_a, vec_b):
    if len(vec_a) != len(vec_b):
        return 0
    return [vec_a[i] * vec_b[i] for i in range(len(vec_a))]

def relu(vec):
    return [x if x > 0 else 0 for x in vec]

def zeros_like(a):
    if isinstance(a[0], list):
        return [[0.0 for _ in row] for row in a]
    return [0.0 for _ in a]

def outer(vec_a, vec_b):
    return [[a * b for b in vec_b]for a in vec_a]

def add_matrix(a, b):
    if len(a) != len(b):
        return 0
    result = []

    for row_a, row_b in zip(a, b):
        if len(row_a) != len(row_b):
            return 0
        result.append([x + y for x, y in zip(row_a, row_b)])
    return result