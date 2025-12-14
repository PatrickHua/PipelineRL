from .template import template
import re
import time

import aiohttp
from omegaconf import DictConfig
from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.llm import Prompt, TrainableLLM
from pipelinerl.rollouts import BaseMetrics, RolloutResult


async def generate_multiply_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    a = problem["a"]
    b = problem["b"]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant",
        },
        {
            "role": "user",
            "content": template.format(a=a, b=b),
        }
    ]
    prompt = Prompt(messages=messages)
    
    
    time_start = time.time()
    
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start
    
    assert llm_call.output.content is not None
    
    rewards = RewardTable(**dict(cfg.rewards))
    discount_factor = cfg.actor.discount_factor
    
    answer = re.search("<answer>(\d+)</answer>", llm_call.output.content)
    if answer:
        answer = int(answer.group(1))
        if answer == a * b:
            reward = rewards.correct_answer
        else:
            reward = rewards.wrong_answer
    return RolloutResult(
        training_texts=[make_training_text(llm, llm_call)],
        metrics=BaseMetrics(
            reward=reward,
            success=success,
            no_error=not error,
            no_answer=error,
        ),
        latency=latency,
        dataset_name=problem["dataset"],
    )