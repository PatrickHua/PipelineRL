from .template import template


def generate_multiply_rollout(cfg: DictConfig, llm: TrainableLLM, problem: dict, session: aiohttp.ClientSession) -> RolloutResult:
    a = problem["a"]
    b = problem["b"]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant",
        },
    ]
    # TODO:
    task_template = template.format(a=a, b=b)
    messages.append({
        "role": "user",
        "content": task_template,
    })
    prompt = Prompt(messages=messages)
    llm_call = await llm_async_generate(llm, prompt, session)
    return llm_call.output.content