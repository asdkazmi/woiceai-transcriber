import math
import os
import json


def cost(handler_duration, execution_time_embeddings):
    embeddings_cost_per_second = float(os.getenv('RUNPOD_API_COST_PER_SECOND'))
    transcriber_cost_per_second = float(os.getenv('RUNPOD_TRANSCRIBER_COST_PER_SECOND'))

    embeddings_cost = embeddings_cost_per_second * execution_time_embeddings
    prompt_builder_cost = handler_duration * transcriber_cost_per_second
    total_credits_cents = math.ceil((embeddings_cost + prompt_builder_cost) * 1000)

    return total_credits_cents


"""
Lambda woice kb
memory 1024	$0.0000000167/ms 
memory 3072	$0.0000000500/ms 

BGE 0.00019/s

Transcriber 0.00031/s

llama 0.00031/s
"""
