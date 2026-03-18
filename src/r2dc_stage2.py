
import argparse
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,  
    TrainerCallback,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from transformers.trainer_utils import set_seed


# ---------------- Dataset Loading ----------------
def load_json_dataset(train_file, valid_file):
    data_files = {"train": train_file, "validation": valid_file}
    return load_dataset("json", data_files=data_files)


# ---------------- Preprocessing ----------------
def preprocess_function(examples, tokenizer, max_input_length):
    # Predict inputs (question only)
    predict_inputs = [f"predict: {q}" for q in examples["question"]]
    targets_labels = examples["label"]


    explain_inputs = [
        f"given label: {l}, explain: {q}" for q, l in zip(examples["question"], examples["label"])
    ]
    targets_rationales = examples["rationale"]


    predict_enc = tokenizer(
        predict_inputs, padding=False, truncation=True, max_length=max_input_length
    )
    explain_enc = tokenizer(
        explain_inputs, padding=False, truncation=True, max_length=max_input_length
    )


    return {
        "predict_input_ids": predict_enc["input_ids"],
        "predict_attention_mask": predict_enc["attention_mask"],
        "predict_labels_text": targets_labels,
        "explain_input_ids": explain_enc["input_ids"],
        "explain_attention_mask": explain_enc["attention_mask"],
        "explain_labels_text": targets_rationales,
        "original_question": examples["question"],
    }


# ---------------- Data Collator ----------------
def make_r2dc_data_collator(tokenizer, eval_mode=False):
    def collator(features):
        if eval_mode:
            # Eval: prediction only for eval loss
            batch = tokenizer.pad(
                [{"input_ids": f["predict_input_ids"],
                  "attention_mask": f["predict_attention_mask"]}
                 for f in features],
                return_tensors="pt"
            )
            labels = tokenizer(
                [f["predict_labels_text"] for f in features],
                padding=True, truncation=True, max_length=256,
                return_tensors="pt"
            )
            labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
            batch["labels"] = labels["input_ids"]
            return batch
        else:
            # Train: multitask (predict + explain)
            predict_batch = [{"input_ids": f["predict_input_ids"],
                              "attention_mask": f["predict_attention_mask"]} for f in features]
            explain_batch = [{"input_ids": f["explain_input_ids"],
                              "attention_mask": f["explain_attention_mask"]} for f in features]
            batch = {
                "predict": tokenizer.pad(predict_batch, return_tensors="pt"),
                "explain": tokenizer.pad(explain_batch, return_tensors="pt"),
                "predict_labels_text": [f["predict_labels_text"] for f in features],
                "explain_labels_text": [f["explain_labels_text"] for f in features],
                "questions": [f["original_question"] for f in features],   ### NEW
            }
            return batch
    return collator





class DelayedEarlyStoppingCallback(TrainerCallback):
    """
    Early stopping but only after a delay (e.g., after warmup + mix steps).
    """
    def __init__(self, patience, warmup_steps, mix_steps, metric_name="eval_loss", greater_is_better=False):
        self.patience = patience
        self.warmup_steps = warmup_steps
        self.mix_steps = mix_steps
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_score = None
        self.counter = 0


    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Only start checking after warmup + mix steps
        start_checking_step = self.warmup_steps + self.mix_steps
        if state.global_step < start_checking_step:
            return control  # skip early stopping during warmup/mix


        metric_val = metrics.get(self.metric_name)
        if metric_val is None:
            return control


        # Determine if we improved
        if self.best_score is None:
            self.best_score = metric_val
            self.counter = 0
        else:
            improved = (metric_val > self.best_score) if self.greater_is_better else (metric_val < self.best_score)
            if improved:
                self.best_score = metric_val
                self.counter = 0
            else:
                self.counter += 1


        if self.counter >= self.patience:
            print(f"Early stopping triggered at step {state.global_step}")
            control.should_training_stop = True
        return control



# ---------------- Logging Callback ----------------
class RationaleLoggingCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, num_samples=3):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples


    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()
        device = args.device
        import random
        samples = random.sample(range(len(self.eval_dataset)), self.num_samples)
        for idx in samples:
            ex = self.eval_dataset[idx]
            question = ex["question"]
            # Predict label
            pred_input = self.tokenizer(f"predict: {question}", return_tensors="pt").to(device)
            pred_ids = model.generate(**pred_input, max_length=50)
            pred_text = self.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
            # Predict rationale conditioned on predicted label
            expl_prompt = f"given label: {pred_text}, explain: {question}"
            expl_input = self.tokenizer(expl_prompt, return_tensors="pt").to(device)
            expl_ids = model.generate(**expl_input, max_length=100)
            expl_text = self.tokenizer.decode(expl_ids[0], skip_special_tokens=True)
            print(f"\n[Sample {idx}]")
            print(f"Predicted Label: {pred_text}")
            print(f"Predicted Rationale: {expl_text}\n")


# ---------------- Custom Trainer ----------------
class R2DC_Trainer(Seq2SeqTrainer):
    def __init__(
        self,
        alpha=0.5,
        alpha_warmup_steps=1000,
        pred_label_transition_steps=3000,
        lambda_cycle=0.1,
        cycle_warmup_steps=1000,
        train_collator=None,
        eval_collator=None,
        tokenizer=None,
        gumbel_tau_start=1.0,
        gumbel_tau_min=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_alpha = alpha
        self.alpha_warmup_steps = alpha_warmup_steps
        self.pred_label_transition_steps = pred_label_transition_steps
        self.train_collator = train_collator
        self.eval_collator = eval_collator
        self.tokenizer = tokenizer
        self.lambda_cycle = lambda_cycle
        self.cycle_warmup_steps = cycle_warmup_steps

        self.gumbel_tau_start = gumbel_tau_start
        self.gumbel_tau_min = gumbel_tau_min

        max_steps = self.args.max_steps
        gumbel_tau_decay = 0.6931 / max(1, max_steps)
        self.gumbel_tau_decay = gumbel_tau_decay

        
        self.confidence_tracker = {
            'values': [],  
            'mean': 0.0,   
            'count': 0,    
            'tau': 0.75,  
            'last_update': 0 
        }

        # stats for adaptive scheduled sampling
        self.ss_stats = {
            "steps": 0,
            "used_pred": 0,
            "total": 0,
            "avg_conf": 0.0,
        }


    def get_current_gumbel_tau(self):
        step = self.state.global_step
        # Exponential decay: tau = max(tau_min, tau_start * exp(-r * step))
        gumbel_tau = self.gumbel_tau_start * torch.exp(torch.tensor(-self.gumbel_tau_decay * step))
        return max(self.gumbel_tau_min, gumbel_tau.item())

    def track_confidence(self, confidences, current_step):
        conf_list = confidences.detach().cpu().numpy().tolist()
        self.confidence_tracker['values'].extend(conf_list)
        
        n = len(conf_list)
        old_count = self.confidence_tracker['count']
        old_mean = self.confidence_tracker['mean']
        new_count = old_count + n
        self.confidence_tracker['mean'] = old_mean + (sum(conf_list) - old_mean * n) / new_count
        self.confidence_tracker['count'] = new_count

    def update_tau(self, current_step):
        if current_step == self.cycle_warmup_steps:
            if self.confidence_tracker['count'] > 0:
                initial_tau = self.confidence_tracker['mean']
                self.confidence_tracker['tau'] = initial_tau
                print(f"\n[Warmup Over] Initial τ calibrated to: {initial_tau:.4f}")
                self._reset_tracker_window(current_step)

        elif current_step > self.cycle_warmup_steps:
            steps_since_warmup = current_step - self.cycle_warmup_steps
            if steps_since_warmup % 1000 == 0:
                if self.confidence_tracker['count'] > 0:
                    new_tau = self.confidence_tracker['mean']
                    self.confidence_tracker['tau'] = new_tau
                    print(f" [Step {current_step}] Window Reset: New τ = {new_tau:.4f}")
                
                self._reset_tracker_window(current_step)

    def _reset_tracker_window(self, current_step):
        self.confidence_tracker['values'] = []
        self.confidence_tracker['mean'] = 0.0
        self.confidence_tracker['count'] = 0
        self.confidence_tracker['last_update'] = current_step


    def get_tau(self, current_step):
        """Get current tau based on training phase"""
        # Phase 1: During cycle warmup, collect confidences
        if current_step < self.cycle_warmup_steps:
            # Don't use threshold yet, but track confidences
            return 1.0  # High value = never use predicted labels
        
        # Phase 2: After warmup, use collected average
        elif current_step == self.cycle_warmup_steps:
            # First step after warmup: use mean of collected confidences
            if len(self.confidence_tracker['values']) > 0:
                initial_tau = self.confidence_tracker['mean']
                self.confidence_tracker['tau'] = initial_tau
                print(f"Initial τ set to {initial_tau:.4f} (from {self.confidence_tracker['count']} samples)")
                return initial_tau
            else:
                # Fallback if no data collected. This shouldn't happen but just in case.
                return 0.75
        
        # Phase 3: Continue with dynamic updates
        else:
            return self.confidence_tracker['tau']
        
    def current_alpha(self):
        step = self.state.global_step
        if step >= self.alpha_warmup_steps:
            return self.target_alpha
        return self.target_alpha * step / max(1, self.alpha_warmup_steps)


    def get_train_dataloader(self):
        self.data_collator = self.train_collator
        return super().get_train_dataloader()


    def get_eval_dataloader(self, eval_dataset=None):
        self.data_collator = self.eval_collator
        return super().get_eval_dataloader(eval_dataset)


    def fraction_predicted_labels(self):
        step = self.state.global_step
        if step < self.alpha_warmup_steps:
            return 0.0
        t = (step - self.alpha_warmup_steps) / max(1, self.pred_label_transition_steps)
        return min(0.9, t)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        device = self.args.device
        current_step = self.state.global_step
        if not model.training:
            return super().compute_loss(model, inputs, return_outputs)


        # ========= 1. Prediction loss (Q -> Label) =========
        pred_inputs = {k: v.to(device) for k, v in inputs["predict"].items()}
        pred_labels = self.tokenizer(
            inputs["predict_labels_text"],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        pred_labels_ids = pred_labels["input_ids"]
        pred_labels_ids[pred_labels_ids == self.tokenizer.pad_token_id] = -100
        pred_out = model(**pred_inputs, labels=pred_labels_ids)


        # ========= 2. Generate labels + confidence for adaptive SS =========
        core = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            gen_out = core.generate(
                **pred_inputs,
                max_length=50,
                output_scores=True,
                return_dict_in_generate=True,
            )


        scores = gen_out.scores  # list of [batch_size, vocab_size]
        sequences = gen_out.sequences  # [batch_size, seq_len]


        token_logps = []
        for t, step_scores in enumerate(scores):
            log_probs = torch.log_softmax(step_scores, dim=-1)
            token_ids = sequences[:, t + 1]  # skip BOS
            token_logps.append(
                torch.gather(log_probs, 1, token_ids.unsqueeze(1)).squeeze(1)
            )
        seq_logp = torch.stack(token_logps, dim=1).mean(dim=1)  # [batch_size]
        confidences = torch.exp(seq_logp)  # [batch_size]


        predicted_labels_text = self.tokenizer.batch_decode(
            gen_out.sequences, skip_special_tokens=True
        )

        self.track_confidence(confidences, current_step)
        tau = self.get_tau(current_step)
        self.update_tau(current_step)
        
        # ========= 3. Explanation loss with adaptive scheduled sampling =========
        frac_pred = self.fraction_predicted_labels()
        target_labels_text = []  # labels used in prompts → also used for cycle targets


        if frac_pred == 0.0:
            expl_inputs_pt = {k: v.to(device) for k, v in inputs["explain"].items()}
            target_labels_text = inputs["predict_labels_text"]
        else:
            questions = inputs["questions"]
            expl_prompts = []
            used_pred_count = 0
            conf_sum = 0.0



            for i, (pl, gold, q) in enumerate(
                zip(predicted_labels_text, inputs["predict_labels_text"], questions)
            ):
                conf = confidences[i]
                conf_sum += conf.item()
                lambda_i_tensor = torch.sigmoid(10 * (conf - tau))
                lambda_i = lambda_i_tensor.item()


                use_pred = (
                    torch.rand(1).item() < frac_pred
                    and torch.rand(1).item() < lambda_i
                )
                if use_pred:
                    used_pred_count += 1
                label_for_prompt = pl if use_pred else gold
                target_labels_text.append(label_for_prompt)


                expl_prompts.append(
                    f"given label: {label_for_prompt}, explain: {q}"
                )


            expl_enc = self.tokenizer(
                expl_prompts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            expl_inputs_pt = {
                "input_ids": expl_enc["input_ids"],
                "attention_mask": expl_enc["attention_mask"],
            }


            batch_size = len(questions)
            self.ss_stats["steps"] += 1
            self.ss_stats["used_pred"] += used_pred_count
            self.ss_stats["total"] += batch_size
            self.ss_stats["avg_conf"] += conf_sum / batch_size


        expl_labels = self.tokenizer(
            inputs["explain_labels_text"],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        expl_labels_ids = expl_labels["input_ids"]
        expl_labels_ids[expl_labels_ids == self.tokenizer.pad_token_id] = -100
        expl_out = model(**expl_inputs_pt, labels=expl_labels_ids)


        # ========= 4. Cycle consistency (Rationale -> Label) =========
        cycle_loss = torch.tensor(0.0, device=device)
        cycle_acc = 0.0

        if self.state.global_step > self.cycle_warmup_steps:
            base_model = model.module if hasattr(model, "module") else model

            curr_gumbel_tau = self.get_current_gumbel_tau()
            soft_probs = F.gumbel_softmax(
                            expl_out.logits, 
                            tau=curr_gumbel_tau, 
                            hard=False, 
                            dim=-1
                        )
        
            embeddings_weight = base_model.get_input_embeddings().weight
            soft_rationale_embeds = torch.matmul(soft_probs, embeddings_weight)


            # prepend "predict: " prefix
            prefix_ids = self.tokenizer(
                "predict: ", return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            with torch.no_grad():
                prefix_embeds = base_model.get_input_embeddings()(prefix_ids).expand(
                    soft_rationale_embeds.shape[0], -1, -1
                )


            full_cycle_embeds = torch.cat(
                [prefix_embeds, soft_rationale_embeds], dim=1
            )


            # targets are the labels actually used in the prompts
            cycle_targets = self.tokenizer(
                target_labels_text,
                padding=True,
                truncation=True,
                max_length=50,
                return_tensors="pt",
            ).to(device)
            cycle_labels_ids = cycle_targets["input_ids"]
            cycle_labels_ids[cycle_labels_ids == self.tokenizer.pad_token_id] = -100


            cycle_out = model(
                inputs_embeds=full_cycle_embeds, labels=cycle_labels_ids
            )
            cycle_loss = cycle_out.loss


            with torch.no_grad():
                cycle_preds = torch.argmax(cycle_out.logits, dim=-1)
                mask = cycle_labels_ids != -100
                cycle_acc = (
                    ((cycle_preds == cycle_labels_ids) & mask)
                    .sum()
                    .float()
                    / mask.sum().float()
                )


        if (self.state.global_step % 500 == 0 and self.state.global_step > self.cycle_warmup_steps):
            used = self.ss_stats["used_pred"]
            total = max(1, self.ss_stats["total"])
            avg_conf = self.ss_stats["avg_conf"] / max(1, self.ss_stats["steps"])
            print(
                f"step={self.state.global_step} | "
                f"frac_pred={frac_pred:.3f} | "
                f"used_pred={used/total:.3f} | "
                f"avg_conf={avg_conf:.3f} | "
                f"tau={tau:.4f} | "
                f"cycle_acc={cycle_acc:.4f} | "
                f"cycle_loss={cycle_loss.item():.4f} | "
                f"gumbel_tau={self.get_current_gumbel_tau():.4f} "
            )


        alpha = self.current_alpha()
        loss = (alpha * pred_out.loss+ (1 - alpha) * expl_out.loss+ self.lambda_cycle * cycle_loss)


        return (loss, {"pred_out": pred_out, "expl_out": expl_out}) if return_outputs else loss




# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--stage1_model_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--alpha_warmup_steps", type=int)
    parser.add_argument("--pred_label_transition_steps", type=int)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    
    args = parser.parse_args()
    set_seed(args.seed)
    print(args.seed)
    
    alpha_warmup_steps = int(0.05 * args.max_steps)
    cycle_warmup_steps = alpha_warmup_steps
    pred_label_transition_steps = int(0.6 * args.max_steps)
    
    print(
        f"alpha warmup steps:{alpha_warmup_steps}, "
        f"mix steps: {pred_label_transition_steps}, "
        f"cycle warmup steps: {cycle_warmup_steps}"
    )
    
    tokenizer = T5Tokenizer.from_pretrained(args.stage1_model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.stage1_model_dir)
    if torch.cuda.is_available():
        model = model.to("cuda")


    raw_datasets = load_json_dataset(args.train_file, args.valid_file)
    tokenized = raw_datasets.map(
        lambda ex: preprocess_function(ex, tokenizer, args.max_input_length),
        batched=True,
        remove_columns=["question", "label", "rationale"],
        num_proc=4,
    )


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # Replace -100 with pad token
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean up
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        # Handle empty predictions
        decoded_preds = [p if p else "unknown" for p in decoded_preds]
        
        # Compute metrics
        acc = accuracy_score(decoded_labels, decoded_preds)
        f1 = f1_score(decoded_labels, decoded_preds, average="macro", zero_division=0)
        return {"accuracy": acc, "f1": f1}


    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_dir=os.path.join(args.output_dir, "logs/"),
        logging_steps=50,
        report_to="tensorboard",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=args.seed,
        bf16=True,
        predict_with_generate=True,
        gradient_accumulation_steps=args.grad_steps,
        save_total_limit=2
    )


    train_collator = make_r2dc_data_collator(tokenizer, eval_mode=False)
    eval_collator = make_r2dc_data_collator(tokenizer, eval_mode=True)


    trainer = R2DC_Trainer(
        alpha=args.alpha,
        alpha_warmup_steps=alpha_warmup_steps,
        cycle_warmup_steps=cycle_warmup_steps,
        model=model,
        tokenizer=tokenizer,  
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        train_collator=train_collator,
        pred_label_transition_steps=pred_label_transition_steps,
        eval_collator=eval_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            DelayedEarlyStoppingCallback(
                patience=5, 
                warmup_steps=alpha_warmup_steps,
                mix_steps=pred_label_transition_steps,
                metric_name="eval_f1",  
                greater_is_better=True,   
            ),
            RationaleLoggingCallback(tokenizer, raw_datasets["validation"], num_samples=3)
        ]


    )


    trainer.train()


if __name__ == "__main__":
    main()