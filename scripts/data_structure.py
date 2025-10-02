import h5py


def h5_tree(val: h5py.Group, prefix: str = "") -> None:

    items = list(val.items())
    for i, (key, item) in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "

        if isinstance(item, h5py.Group):
            print(f"{prefix}{connector}{key}")
            new_prefix = prefix + ("    " if is_last else "│   ")
            h5_tree(item, new_prefix)
        else:  # It's a Dataset
            info = ""
            if item.ndim == 0:  # scalar dataset
                try:
                    value = item.asstr()[()] if h5py.check_string_dtype(item.dtype) else item[()]
                    info = f"({value!r})"
                except (TypeError, AttributeError):
                    info = f"(scalar value: {item[()]})"
            else:  # Array dataset
                info = f"(shape: {item.shape}, dtype: {item.dtype})"

            print(f"{prefix}{connector}{key} {info}")




if __name__ == "__main__":

    file_name = r"D:\03_Resources\Data\XRR_AI\data\250929.h5"

    # 생성된 파일 구조 출력
    print(f"< HDF5 file: \"{file_name}\" >")
    with h5py.File(file_name, 'r') as hf:
        h5_tree(hf)
