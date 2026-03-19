import os
import random
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dist_is_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank_world():
    if dist_is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1

def dist_init():
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

def barrier():
    if dist_is_initialized():
        dist.barrier()

def all_gather_list(obj_list_local):
    if not dist_is_initialized():
        return obj_list_local
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj_list_local)
    merged = []
    for part in gathered:
        merged.extend(part)
    return merged

def normalize_text_list(xs):
    return [str(x).strip().lower() for x in xs]

def main():
    set_seed(42)
    dist_init()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank, world = get_rank_world()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    model_dir = "" #path to the stage 2 model
    test_csv = "" #path to the test CSV file with columns: question, label, rationale
    output_csv = "" #path to the output CSV file

    question_col = "question"
    label_col = "label"
    rationale_col = "rationale"

    max_input_length = 1024
    max_output_length_label = 50
    max_output_length_rationale = 256
    batch_size = 64

    if rank == 0:
        print(f"Using device: {device}, world_size={world}")

    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()

    df = pd.read_csv(test_csv)
    questions = df[question_col].astype(str).tolist()
    gold_labels = df[label_col].astype(str).tolist()
    gold_rationales = df[rationale_col].astype(str).tolist()

    idx_all = list(range(len(df)))
    idx_local = idx_all[rank::world]

    def take(lst, idxs):
        return [lst[i] for i in idxs]

    local_questions = take(questions, idx_local)
    local_gold_labels = take(gold_labels, idx_local)
    local_gold_rationales = take(gold_rationales, idx_local)

    # Step 1: P1 from question
    p1_local = []
    with torch.no_grad():
        for i in tqdm(range(0, len(local_questions), batch_size), desc=f"R{rank} P1", disable=(rank != 0)):
            batch_q = local_questions[i:i + batch_size]
            prompts = [f"predict: {q}" for q in batch_q]
            enc = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt"
            ).to(device)
            out = model.generate(**enc, max_length=max_output_length_label, temperature=0.0)
            p1_local.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    p1_pairs = all_gather_list(list(zip(idx_local, p1_local)))
    p1_pairs.sort(key=lambda x: x[0])
    p1_all = [x[1] for x in p1_pairs]

    # Step 2: Generated rationale from question + conditioned on P1
    gen_rat_local = []
    local_p1 = take(p1_all, idx_local)
    with torch.no_grad():
        prompts_all = [f"given label: {p}, explain: {q}" for q, p in zip(local_questions, local_p1)]
        for i in tqdm(range(0, len(prompts_all), batch_size), desc=f"R{rank} RAT", disable=(rank != 0)):
            batch = prompts_all[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt"
            ).to(device)
            out = model.generate(**enc, max_length=max_output_length_rationale, temperature=0.0)
            gen_rat_local.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    rat_pairs = all_gather_list(list(zip(idx_local, gen_rat_local)))
    rat_pairs.sort(key=lambda x: x[0])
    gen_rat_all = [x[1] for x in rat_pairs]

    # Step 3: P2 from generated rationale
    p2_local = []
    local_gen_rat = take(gen_rat_all, idx_local)
    with torch.no_grad():
        prompts_all = [f"predict: {r}" for r in local_gen_rat]
        for i in tqdm(range(0, len(prompts_all), batch_size), desc=f"R{rank} P2", disable=(rank != 0)):
            batch = prompts_all[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt"
            ).to(device)
            out = model.generate(**enc, max_length=max_output_length_label, temperature=0.0)
            p2_local.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    p2_pairs = all_gather_list(list(zip(idx_local, p2_local)))
    p2_pairs.sort(key=lambda x: x[0])
    p2_all = [x[1] for x in p2_pairs]

    # Step 4: Pg from gold rationale
    pg_local = []
    with torch.no_grad():
        prompts_all = [f"predict: {r}" for r in local_gold_rationales]
        for i in tqdm(range(0, len(prompts_all), batch_size), desc=f"R{rank} PG", disable=(rank != 0)):
            batch = prompts_all[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt"
            ).to(device)
            out = model.generate(**enc, max_length=max_output_length_label, temperature=0.0)
            pg_local.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    pg_pairs = all_gather_list(list(zip(idx_local, pg_local)))
    pg_pairs.sort(key=lambda x: x[0])
    pg_all = [x[1] for x in pg_pairs]

    barrier()

    if rank == 0:
        gold_norm = normalize_text_list(gold_labels)
        p1_norm = normalize_text_list(p1_all)
        p2_norm = normalize_text_list(p2_all)
        pg_norm = normalize_text_list(pg_all)

        label_f1 = f1_score(gold_norm, p1_norm, average="macro")
        consistency = accuracy_score(p1_norm, p2_norm)
        sufficiency = accuracy_score(gold_norm, p2_norm)
        gold_robustness = accuracy_score(gold_norm, pg_norm)

        print("\n" + "=" * 50)
        print("FINAL EVALUATION RESULTS")
        print(f"Label Macro F1        : {label_f1:.4f}")
        print(f"Consistency (P1==P2)  : {consistency:.4f}")
        print(f"Sufficiency (P2==GT)  : {sufficiency:.4f}")
        print(f"Gold Robustness       : {gold_robustness:.4f}")
        print("=" * 50)

        print("\n=== Sample predictions ===\n")
        for i in range(min(10, len(df))):
            print(f"SAMPLE {i+1}")
            print(f"Question           : {questions[i][:300]}")
            print(f"Gold Label         : {gold_labels[i]}")
            print(f"P1 Pred Label      : {p1_all[i]}")
            print(f"Gold Rationale     : {gold_rationales[i][:300]}")
            print(f"Generated Rationale: {gen_rat_all[i][:300]}")
            print(f"P2 from Gen Rat    : {p2_all[i]}")
            print(f"Pg from Gold Rat   : {pg_all[i]}")
            print("-" * 60)

        output_df = pd.DataFrame({
            question_col: questions,
            label_col: gold_labels,
            rationale_col: gold_rationales,
            "P1_Predicted_Label": p1_all,
            "Generated_Rationale": gen_rat_all,
            "P2_From_Generated_Rationale": p2_all,
            "Pg_From_Gold_Rationale": pg_all
        })
        output_df.to_csv(output_csv, index=False)
        print(f"\nSaved CSV to: {output_csv}")

if __name__ == "__main__":
    main()
