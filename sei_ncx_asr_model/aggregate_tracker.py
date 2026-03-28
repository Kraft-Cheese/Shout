#!/usr/bin/env python3
import json
import csv
from pathlib import Path

def parse_metrics(json_path):
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)

def run_aggregation():
    output_rows = []
    
    # Define files layout based on B1 - B4
    configs = [
        {"id": "B1", "lang": "ncx", "metric_file": "b1_results/b1_zero_shot_metrics.json", "recon": "off"},
        {"id": "B1", "lang": "sei", "metric_file": "b1_results/b1_zero_shot_metrics.json", "recon": "off"},
        {"id": "B2", "lang": "sei", "metric_file": "sei_lora_adapter/test_metrics.json", "recon": "off"},
        {"id": "B2", "lang": "ncx", "metric_file": "ncx_lora_adapter/test_metrics.json", "recon": "off"},
        {"id": "B3", "lang": "ncx/sei", "metric_file": "ncx_asr_model/ncx-sei-joint-asr/train_metrics.json", "recon": "off"},
        {"id": "B4", "lang": "ncx", "metric_file": "b4_results/b4_ncx_metrics.json", "recon": "on"},
        {"id": "B4", "lang": "sei", "metric_file": "b4_results/b4_sei_metrics.json", "recon": "on"},
    ]
    
    for cfg in configs:
        m = parse_metrics(Path(cfg["metric_file"]))
        
        row = {
            "Baseline ID": cfg["id"],
            "Language": cfg["lang"],
            "Model": "Whisper",
            "Dataset": "Common Voice / Fallback",
            "Reconstruction": cfg["recon"],
            "WER": None,
            "CER": None,
            "Latency": None,
            "Memory": None
        }
        
        if m:
            if isinstance(m, list):
                if len(m) > 0:
                    m = m[0]
                else:
                    m = {}
            # Handle different JSON key structures across B1-B4
            if "test_wer" in m:
                row["WER"] = round(m["test_wer"], 4)
                row["CER"] = round(m.get("test_cer", 0.0), 4)
            elif "wer" in m:
                row["WER"] = round(m["wer"], 4)
                row["CER"] = round(m.get("cer", 0.0), 4)
            elif "eval_wer" in m: # For training metrics
                row["WER"] = round(m["eval_wer"], 4)
                row["CER"] = round(m.get("eval_cer", 0.0), 4)
            
            # Additional B1 specifics
            if cfg["id"] == "B1" and cfg["lang"] == "ncx" and "ncx" in m:
                row["WER"] = round(m["ncx"].get("wer", 0.0), 4)
                row["CER"] = round(m["ncx"].get("cer", 0.0), 4)
            if cfg["id"] == "B1" and cfg["lang"] == "sei" and "sei" in m:
                row["WER"] = round(m["sei"].get("wer", 0.0), 4)
                row["CER"] = round(m["sei"].get("cer", 0.0), 4)
            
            # Runtime 
            row["Latency"] = round(m.get("test_runtime", m.get("eval_runtime", 0.0)), 2)
            
        output_rows.append(row)

    # Write CSV
    with open("experiment_tracker.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
        
    print("experiment_tracker.csv created.")

if __name__ == "__main__":
    run_aggregation()
