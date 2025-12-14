

def generate_countdown_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:

    # Add system prompt if specified
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    
    question_text = cfg.actor.task_template.format(question=problem["question"])
    
    return RolloutResult(
        training_texts=[training_text],
        metrics=metrics,
    )




def load_problems(dataset_names: list[str]):
    problems = []
    for name in dataset_names:
        if name == "train":
            problems.extend([
                {"answer": (2 * i * c) % n + 1, "dataset": "train"} for i in range(512)
            ])
        elif name == "test":
            problems.extend([
                {"answer": ((2 * i + 1) * c) % n + 1, "dataset": "test"} for i in range(512)
            ])
    return problems