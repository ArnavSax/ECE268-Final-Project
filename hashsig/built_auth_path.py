def build_auth_path(
    tree: np.ndarray,
    leaf_index: int,
    height: int,
    hash_size: int = 32
) -> bytes:
    path = []

    node = (1 << height) + leaf_index

    while node > 1:
        sibling = node ^ 1

        path.append(
            tree[
                sibling * hash_size:
                (sibling + 1) * hash_size
            ]
        )

        node >>= 1

    return b"".join(path)
