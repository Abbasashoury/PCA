from src.Data_loader import DataLoader
from src.Center import center


def main():
    print("=== Step 1: Load data ===")
    loader = DataLoader()
    X, t = loader.load()
    loader.get_info()

    print("\n=== Step 2: Center data ===")
    Center = center()
    B = Center.center(X)
    Center.verify_centering(B)


if __name__ == '__main__':
    main()
