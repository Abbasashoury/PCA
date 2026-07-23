from src.Data_loader import DataLoader

def main():
    print("=== Step 1: Load data ===")
    loader = DataLoader()
    X, t = loader.load()
    loader.get_info()

if __name__ == '__main__':
    main()