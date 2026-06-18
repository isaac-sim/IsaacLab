#!/usr/bin/env python3
import argparse
import h5py
import numpy as np


def print_attrs(obj, indent=""):
    """Print HDF5 attributes."""
    if len(obj.attrs) > 0:
        print(f"{indent}Attributes:")
        for key, value in obj.attrs.items():
            print(f"{indent}  - {key}: {value}")


def preview_dataset(dataset, indent="", max_elements=20):
    """Print a small preview of dataset contents."""
    try:
        data = dataset[()]

        if np.isscalar(data):
            print(f"{indent}Value: {data}")
            return

        flat = np.array(data).flatten()

        if flat.size == 0:
            print(f"{indent}Preview: empty dataset")
            return

        preview = flat[:max_elements]
        print(f"{indent}Preview first {min(max_elements, flat.size)} values:")
        print(f"{indent}  {preview}")

        if flat.size > max_elements:
            print(f"{indent}  ... total elements: {flat.size}")

    except Exception as e:
        print(f"{indent}Preview failed: {e}")


def inspect_item(name, obj, max_preview_elements=20):
    """Callback used by h5py visititems()."""
    depth = name.count("/")
    indent = "  " * depth

    if isinstance(obj, h5py.Group):
        print(f"{indent}[Group] {name}/")
        print_attrs(obj, indent + "  ")

    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}[Dataset] {name}")
        print(f"{indent}  Shape: {obj.shape}")
        print(f"{indent}  Dtype: {obj.dtype}")
        print(f"{indent}  Size: {obj.size}")
        print(f"{indent}  Compression: {obj.compression}")
        print_attrs(obj, indent + "  ")

        # Only preview reasonably small or partially readable datasets
        if obj.size <= max_preview_elements:
            preview_dataset(obj, indent + "  ", max_preview_elements)
        else:
            try:
                preview = obj[tuple(slice(0, min(s, 5)) for s in obj.shape)]
                flat = np.array(preview).flatten()[:max_preview_elements]
                print(f"{indent}  Preview:")
                print(f"{indent}    {flat}")
            except Exception as e:
                print(f"{indent}  Preview failed: {e}")


def inspect_hdf5(file_path, max_preview_elements=20):
    with h5py.File(file_path, "r") as f:
        print(f"HDF5 file: {file_path}")
        print("=" * 80)

        print_attrs(f, "")

        if len(f.keys()) == 0:
            print("File is empty.")
            return

        f.visititems(lambda name, obj: inspect_item(name, obj, max_preview_elements))


def main():
    parser = argparse.ArgumentParser(description="Inspect the structure and contents of an HDF5 file.")
    parser.add_argument("file", help="Path to the HDF5 file")
    parser.add_argument(
        "--max-preview-elements",
        type=int,
        default=20,
        help="Maximum number of values to preview per dataset",
    )

    args = parser.parse_args()
    inspect_hdf5(args.file, args.max_preview_elements)


if __name__ == "__main__":
    main()
