import glob
import json
import os


def main():
    # 1. Find the latest vram_bench_*.json
    bench_files = glob.glob('output/vram_bench_*.json')
    if not bench_files:
        print("No vram_bench_*.json found.")
        return
    latest_bench_file = max(bench_files, key=os.path.getmtime)

    # 2. Load the files
    with open(latest_bench_file, 'r', encoding='utf-8') as f:
        bench_data = json.load(f)

    with open('output/model_stats.json', 'r', encoding='utf-8') as f:
        model_stats = json.load(f)

    # 3. Process the data
    results = bench_data.get('results', [])
    meta = bench_data.get('meta', {})

    # Group by model
    models_data = {}
    for r in results:
        model = r['model']
        if model not in models_data:
            models_data[model] = []
        models_data[model].append(r)

    # 4. Generate Markdown
    md = [
        "# VRAM Benchmark Analysis",
        "",
        f"**Source File**: `{os.path.basename(str(latest_bench_file))}`",
        f"**Date**: {meta.get('timestamp', 'Unknown')}",
        f"**Host**: {meta.get('host', 'Unknown')}",
        "",
        "## Model Summary",
        "",
        "| Model | Params | Quant | Family | Max VRAM Delta (GB) | Avg Prefill t/s | Avg Decode t/s | Prefill Var | Decode Var | Errors |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    summary_rows = []

    for model, stats in models_data.items():
        ms = model_stats.get(model, {})
        params = ms.get('parameter_size', 'N/A')
        quant = ms.get('quantization_level', 'N/A')
        family = ms.get('family', 'N/A')
        
        max_delta = 0
        
        pre_list = []
        dec_list = []
        errors = 0
        
        for r in stats:
            if r.get('error'):
                errors += 1
                
            delta_dict = r.get('vram_delta_mb') or {}
            sum_delta = sum(delta_dict.values())
            
            if sum_delta > max_delta:
                max_delta = sum_delta
            
            pre = r.get('prefill_tokens_per_sec')
            dec = r.get('decode_tokens_per_sec')
            
            if pre is not None:
                pre_list.append(pre)
            if dec is not None:
                dec_list.append(dec)
            
        pre_avg = sum(pre_list) / len(pre_list) if pre_list else 0
        dec_avg = sum(dec_list) / len(dec_list) if dec_list else 0
        
        pre_var = f"{((max(pre_list) - min(pre_list)) / min(pre_list) * 100):.1f}%" if len(pre_list) > 1 and min(pre_list) > 0 else "-"
        dec_var = f"{((max(dec_list) - min(dec_list)) / min(dec_list) * 100):.1f}%" if len(dec_list) > 1 and min(dec_list) > 0 else "-"
        
        max_delta_gb = max_delta / 1024.0
            
        summary_rows.append({
            'model': model,
            'params': params,
            'quant': quant,
            'family': family,
            'max_delta_gb': max_delta_gb,
            'pre_avg': pre_avg,
            'dec_avg': dec_avg,
            'pre_var': pre_var,
            'dec_var': dec_var,
            'errors': errors
        })

    # Sort by Max VRAM Delta (GB) descending
    summary_rows.sort(key=lambda x: x['max_delta_gb'], reverse=True)

    for row in summary_rows:
        md.append(f"| {row['model']} | {row['params']} | {row['quant']} | {row['family']} | {row['max_delta_gb']:.1f} | {row['pre_avg']:.1f} | {row['dec_avg']:.1f} | {row['pre_var']} | {row['dec_var']} | {row['errors']} |")

    md.append("")
    md.append("## Detailed Results")
    md.append("")

    for model, stats in models_data.items():
        md.append(f"### {model}")
        md.append("")
        md.append("| Context | VRAM Delta (GB) | VRAM Total (GB) | Prefill t/s | Decode t/s | Status |")
        md.append("|---|---|---|---|---|---|")
        
        # Sort by ctx
        stats.sort(key=lambda x: x.get('ctx', 0))
        for r in stats:
            ctx = r.get('ctx')
            
            delta_dict = r.get('vram_delta_mb') or {}
            total_dict = r.get('vram_total_mb') or {}
            sum_delta_gb = sum(delta_dict.values()) / 1024.0
            sum_total_gb = sum(total_dict.values()) / 1024.0
            
            pre = r.get('prefill_tokens_per_sec')
            dec = r.get('decode_tokens_per_sec')
            
            pre_str = f"{pre:.1f}" if pre else "-"
            dec_str = f"{dec:.1f}" if dec else "-"
            
            status = r.get('error') or "OK"
            
            md.append(f"| {ctx} | {sum_delta_gb:.1f} | {sum_total_gb:.1f} | {pre_str} | {dec_str} | {status} |")
        
        md.append("")

    output_path = 'docs/vram_bench_report.md'
    os.makedirs('docs', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f"Report generated successfully at {output_path}")

if __name__ == '__main__':
    main()