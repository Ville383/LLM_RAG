# Modified from: onnxruntime-genai/examples/python/phi3-qa.py
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import onnxruntime_genai as og

from vector_database import VectorDatabase
from monitoring import HardwareMonitor
import statistics

# For benchmarking stats
def calc_stats(values):
    return {
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0,
        'p90': statistics.quantiles(values, n=100, method="inclusive")[89] if len(values) >= 5 else 0,
        'p95': statistics.quantiles(values, n=100, method="inclusive")[94] if len(values) >= 5 else 0,
        'p99': statistics.quantiles(values, n=100, method="inclusive")[98] if len(values) >= 5 else 0,
    }

def main(args):
    monitor = HardwareMonitor(interval=0.1)
    monitor.start()

    # Initialize vector database
    db = VectorDatabase()
    db.load_or_create('documents/_pokedex_knowledge_base.md', save_path='vector_db', verbose=False)

    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            config.append_provider(args.execution_provider)
    
    t0 = time.perf_counter() # Model load time

    model = og.Model(config)

    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()

    search_options = {
        name: getattr(args, name)
        for name in ["do_sample", "max_length", "min_length", "top_p", "top_k", "temperature", "repetition_penalty"]
        if name in args
    }

    # Set the max length to something sensible by default
    if "max_length" not in search_options:
        search_options["max_length"] = 2048
    
    load_time_s = time.perf_counter() - t0

    # Load benchmark questions from file
    if args.questions_file:
        questions_path = Path(args.questions_file) 
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"No question bank found in {args.questions_file}.")

    # Total iterations: warmup + benchmark
    results = []
    for iteration in range(args.warmup_rounds + args.benchmark_rounds):
        text = questions[iteration % len(questions)] # Query
        
        # RAG
        started_timestamp_rag = time.perf_counter()

        retrieved_context = db.retrieve(text)

        started_timestamp = time.perf_counter()
        
        # Create prompt
        input_message = [
            {
                "role": "system", 
                "content": "Answer the user's Question based solely on the provided Context. If the answer cannot be found in the Context, refuse to give an answer."
            },
            {
                "role": "user", 
                "content": f"Context:\n{retrieved_context}\n\nQuestion:\n{text}"
            }
        ]
        input_prompt = tokenizer.apply_chat_template(json.dumps(input_message), add_generation_prompt=True)
        input_tokens = tokenizer.encode(input_prompt)

        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)
        generator = og.Generator(model, params)
        generator.append_tokens(input_tokens)

        first_token = True
        n_out_tokens = 0
        try:
            while True:
                generator.generate_next_token()
                if first_token:
                    first_token_timestamp = time.perf_counter()
                    first_token = False
                
                if generator.is_done():
                    break
 
                new_token = generator.get_next_tokens()[0] # batch size 1
                _ = tokenizer_stream.decode(new_token)
                n_out_tokens += 1
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        
        stop_timestamp = time.perf_counter()
        
        # Calculate metrics
        TTFT = first_token_timestamp - started_timestamp
        TPS_input = len(input_tokens) / TTFT
        if n_out_tokens > 0:
            TPS_output = n_out_tokens / (stop_timestamp - first_token_timestamp)
        else:
            TPS_output = 0.0
        
        if iteration >= args.warmup_rounds:
            result = {
                'iteration': iteration + 1,
                'rag_time': started_timestamp - started_timestamp_rag,
                'input_tokens': len(input_tokens),
                'output_tokens': n_out_tokens,
                'time_to_first_token': TTFT,
                'input_tokens_per_second': TPS_input,
                'output_tokens_per_second': TPS_output,
            }
            results.append(result)

    monitor.stop() # stop monitoring, join thread
    monitor.save_plot(args.execution_provider)
    hw_summary = monitor.get_stats()
    
    if results:
        stats = {
            'rag_time': calc_stats(
                [r['rag_time'] for r in results]
            ),
            'time_to_first_token': calc_stats(
                [r['time_to_first_token'] for r in results]
            ),
            'output_tokens_per_second': calc_stats(
                [r['output_tokens_per_second'] for r in results]
            ),
            'input_tokens_per_second': calc_stats(
                [r['input_tokens_per_second'] for r in results]
            ),
            'input_tokens': [r['input_tokens'] for r in results],
            'output_tokens': [r['output_tokens'] for r in results],
            'model_load_time': load_time_s,
        }
    else:
        stats = {}
    
    # Save results
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_path': args.model_path,
            'execution_provider': args.execution_provider,
            'warmup_rounds': args.warmup_rounds,
            'benchmark_rounds': args.benchmark_rounds,
            'search_options': search_options,
        },
        'statistics': stats,
        'hardware': hw_summary,
    }
    output_file = Path("results", f"benchmark_{args.execution_provider}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Total rounds: {args.warmup_rounds} warmup + {args.benchmark_rounds} benchmark")
    print(f"Results saved to: {output_file}")
    
    if stats:
        print(f"\nMetrics (across {len(results)} benchmark rounds):")
        r = stats['rag_time']
        print(f"RAG Retrieval Time: {r['mean']:.3f}s (±{r['stdev']:.3f}s) | "
            f"median: {r['median']:.3f}s, p90: {r['p90']:.3f}s, p95: {r['p95']:.3f}s, p99: {r['p99']:.3f}s")
        t = stats['time_to_first_token']
        print(f"Time to First Token: {t['mean']:.3f}s (±{t['stdev']:.3f}s) | "
            f"median: {t['median']:.3f}s, p90: {t['p90']:.3f}s, p95: {t['p95']:.3f}s, p99: {t['p99']:.3f}s")
        g = stats['output_tokens_per_second']
        print(f"Generation Speed: {g['mean']:.2f} tok/s (±{g['stdev']:.2f}) | "
            f"median: {g['median']:.2f}, p90: {g['p90']:.2f}, p95: {g['p95']:.2f}, p99: {g['p99']:.2f}")
        p = stats['input_tokens_per_second']
        print(f"Prompt Processing: {p['mean']:.2f} tok/s (±{p['stdev']:.2f}) | "
            f"median: {p['median']:.2f}, p90: {p['p90']:.2f}, p95: {p['p95']:.2f}, p99: {p['p99']:.2f}")
        print(f"Model Load Time: {stats['model_load_time']:.2f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai"
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Onnx model folder path (must contain genai_config.json and model.onnx)",
    )
    parser.add_argument(
        "-e",
        "--execution_provider",
        type=str,
        required=False,
        default="follow_config",
        choices=["cpu", "cuda", "dml", "NvTensorRtRtx", "follow_config"],
        help="Execution provider to run the ONNX Runtime session with. Defaults to follow_config that uses the execution provider listed in the genai_config.json instead.",
    )
    parser.add_argument("-i", "--min_length", type=int, help="Min number of tokens to generate including the prompt")
    parser.add_argument("-l", "--max_length", type=int, help="Max number of tokens to generate including the prompt")
    parser.add_argument(
        "-ds",
        "--do_sample",
        action="store_true",
        default=False,
        help="Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false",
    )
    parser.add_argument("-p", "--top_p", type=float, help="Top p probability to sample with")
    parser.add_argument("-k", "--top_k", type=int, help="Top k tokens to sample from")
    parser.add_argument("-t", "--temperature", type=float, help="Temperature to sample with")
    parser.add_argument("-r", "--repetition_penalty", type=float, help="Repetition penalty to sample with")
    
    # --- Benchmark arguments ---
    parser.add_argument(
        "-w",
        "--warmup_rounds",
        type=int,
        default=2,
        help="Number of warmup rounds before benchmark (default: 2)"
    )
    parser.add_argument(
        "-n",
        "--benchmark_rounds",
        type=int,
        default=30,
        help="Number of benchmark rounds to run (default: 30)"
    )
    parser.add_argument(
        "-q",
        "--questions_file",
        type=str,
        default="question_bank.txt",
        help="Path to text file containing questions (one per line)"
    )
    args = parser.parse_args()
    
    main(args)