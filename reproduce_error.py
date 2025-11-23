
try:
    from plotting import plot_prediction
    print("Import successful")
except NameError as e:
    print(f"Caught expected error: {e}")
except Exception as e:
    print(f"Caught unexpected error: {e}")
