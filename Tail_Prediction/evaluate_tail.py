import csv
import os

import numpy as np
import torch


def _rank_from_logits(logits, true_label):
    order = np.argsort(-logits)
    rank = int(np.where(order == true_label)[0][0]) + 1
    return order, rank


def evaluate_model(
    model,
    dataloader,
    device,
    label_list,
    save_dir=None,
    split_name="test",
):
    model.eval()
    ranks = []
    top1 = []
    top3 = []
    top10 = []
    records = []

    with torch.no_grad():
        for inputs, labels, meta in dataloader:
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = labels.detach().cpu().numpy()

            # meta may be a dict of lists (collated mapping) or a list of dicts
            if isinstance(meta, dict):
                heads = meta.get("head")
                relations = meta.get("relation")
                tails = meta.get("tail")
            else:
                heads = [m["head"] for m in meta]
                relations = [m["relation"] for m in meta]
                tails = [m["tail"] for m in meta]

            for idx, logit_row in enumerate(logits):
                order, rank = _rank_from_logits(logit_row, int(label_ids[idx]))
                ranks.append(rank)

                top1_label = label_list[int(order[0])]
                top3_labels = [label_list[int(i)] for i in order[:3]]
                top10_labels = [label_list[int(i)] for i in order[:10]]

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

    ranks_array = np.array(ranks, dtype=np.float32)
    metrics = {
        "MRR": float(np.mean(1.0 / ranks_array)),
        "Hits@1": float(np.mean(ranks_array <= 1)),
        "Hits@3": float(np.mean(ranks_array <= 3)),
        "Hits@5": float(np.mean(ranks_array <= 5)),
        "Hits@10": float(np.mean(ranks_array <= 10)),
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
