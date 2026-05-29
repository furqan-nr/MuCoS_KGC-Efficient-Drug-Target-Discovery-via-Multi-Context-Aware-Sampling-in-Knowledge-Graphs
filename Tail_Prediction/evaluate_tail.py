import csv
import os

import torch


def evaluate_model(
    model,
    dataloader,
    device,
    label_list,
    save_dir=None,
    split_name="test",
    relation_prior=None,
    prior_alpha=0.0,
):
    model.eval()
    ranks = []
    top1 = []
    top3 = []
    top10 = []
    records = []

    prior_bias = None
    if relation_prior:
        label_to_idx = {label: idx for idx, label in enumerate(label_list)}
        prior_bias = {}
        for relation, tail_scores in relation_prior.items():
            bias_vec = torch.zeros(len(label_list), dtype=torch.float32)
            for tail, log_prob in tail_scores.items():
                idx = label_to_idx.get(tail)
                if idx is not None:
                    bias_vec[idx] = float(log_prob)
            prior_bias[relation] = bias_vec

    with torch.no_grad():
        for inputs, labels, meta in dataloader:
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu()
            label_ids = labels.detach().cpu()

            # meta may be a dict of lists (collated mapping) or a list of dicts
            if isinstance(meta, dict):
                heads = meta.get("head")
                relations = meta.get("relation")
                tails = meta.get("tail")
            else:
                heads = [m["head"] for m in meta]
                relations = [m["relation"] for m in meta]
                tails = [m["tail"] for m in meta]

            if prior_bias and prior_alpha:
                adjusted = logits.clone()
                for row_idx in range(adjusted.size(0)):
                    relation = relations[row_idx]
                    bias_vec = prior_bias.get(relation)
                    if bias_vec is not None:
                        adjusted[row_idx] = adjusted[row_idx] + (prior_alpha * bias_vec)
                logits = adjusted

            true_scores = logits.gather(1, label_ids.view(-1, 1))
            batch_ranks = 1 + (logits > true_scores).sum(dim=1)
            ranks.extend(batch_ranks.tolist())

            max_k = min(10, logits.size(1))
            topk = torch.topk(logits, k=max_k, dim=1).indices

            for idx, topk_row in enumerate(topk):
                topk_ids = topk_row.tolist()
                rank = int(batch_ranks[idx])
                top1_label = label_list[int(topk_ids[0])]
                top3_labels = [label_list[int(i)] for i in topk_ids[: min(3, len(topk_ids))]]
                top10_labels = [label_list[int(i)] for i in topk_ids[: min(10, len(topk_ids))]]

                top1.append(top1_label)
                top3.append(top3_labels)
                top10.append(top10_labels)

                records.append(
                    {
                        "head": heads[idx],
                        "relation": relations[idx],
                        "true_tail": tails[idx],
                        "true_tail_rank": rank,
                        "top1": top1_label,
                        "top3": "|".join(top3_labels),
                        "top10": "|".join(top10_labels),
                    }
                )

    ranks_array = torch.tensor(ranks, dtype=torch.float32)
    metrics = {
        "MRR": float(torch.mean(1.0 / ranks_array).item()) if len(ranks_array) else 0.0,
        "Hits@1": float(torch.mean((ranks_array <= 1).float()).item()) if len(ranks_array) else 0.0,
        "Hits@3": float(torch.mean((ranks_array <= 3).float()).item()) if len(ranks_array) else 0.0,
        "Hits@5": float(torch.mean((ranks_array <= 5).float()).item()) if len(ranks_array) else 0.0,
        "Hits@10": float(torch.mean((ranks_array <= 10).float()).item()) if len(ranks_array) else 0.0,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        predictions_path = os.path.join(save_dir, f"{split_name}_predictions.csv")
        ranks_path = os.path.join(save_dir, f"{split_name}_ranks.csv")

        with open(predictions_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "head",
                    "relation",
                    "true_tail",
                    "true_tail_rank",
                    "top1",
                    "top3",
                    "top10",
                ],
            )
            writer.writeheader()
            writer.writerows(records)

        with open(ranks_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["head", "relation", "true_tail", "true_tail_rank"],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "head": record["head"],
                        "relation": record["relation"],
                        "true_tail": record["true_tail"],
                        "true_tail_rank": record["true_tail_rank"],
                    }
                )

    return metrics
