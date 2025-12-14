import random



def load_multiply_problems(
    dataset_names: list[str],
    min_value: int = 1,
    max_value: int = 100,
    train_size: int = 1000,
    test_size: int = 100,
    seed: int = 42,
) -> list[dict]:
    # Set seed for reproducibility
    random.seed(seed)
    
    problems = []
    for name in dataset_names:
        if name == "train":
            problems.extend([
                {
                    "a": random.randint(min_value, max_value), 
                    "b": random.randint(min_value, max_value),
                    "dataset": name
                } for i in range(train_size)
            ])
        elif name == "test":
            problems.extend([
                {
                    "a": random.randint(min_value, max_value), 
                    "b": random.randint(min_value, max_value), 
                    "dataset": name
                } for i in range(test_size)
            ])
        else:
            raise ValueError(f"Invalid dataset name: {name}")
    return problems