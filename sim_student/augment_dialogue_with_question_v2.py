import argparse
from pathlib import Path

import pandas as pd


USAGE_COLS = [
    "UserId",
    "QuestionId",
    "DateAnswered",
    "QuizSessionId",
    "QuizId",
    "QuestionType",
    "IsCorrect",
]


def load_inputs(interventions_path: str, usage_path: str):
    df_interv = pd.read_csv(interventions_path)
    df_usage = pd.read_csv(usage_path)
    set_trace()

    df_interv["FirstMessage"] = pd.to_datetime(df_interv["FirstMessage"])
    df_usage["DateAnswered"] = pd.to_datetime(df_usage["DateAnswered"])
    df_interv["InterventionEnd"] = (
        df_interv["FirstMessage"]
        + pd.to_timedelta(df_interv["InterventionSeconds"], unit="s")
    )

    return df_interv, df_usage



def get_prev_next_matches(df_interv: pd.DataFrame, df_usage: pd.DataFrame) -> pd.DataFrame:
    set_trace()
    left = df_interv.sort_values(["FirstMessage", "UserId"]).reset_index(drop=True).copy()
    right = df_usage.sort_values(["DateAnswered", "UserId"]).reset_index(drop=True).copy()

    prev_match = pd.merge_asof(
        left,
        right[USAGE_COLS],
        left_on="FirstMessage",
        right_on="DateAnswered",
        by="UserId",
        direction="backward",
        allow_exact_matches=True,
    )

    next_match = pd.merge_asof(
        left,
        right[USAGE_COLS],
        left_on="FirstMessage",
        right_on="DateAnswered",
        by="UserId",
        direction="forward",
        allow_exact_matches=True,
    )

    out = left.copy()
    for prefix, match in (("Prev", prev_match), ("Next", next_match)):
        for col in ["QuestionId", "DateAnswered", "QuizSessionId", "QuizId", "QuestionType", "IsCorrect"]:
            out[f"{prefix}{col}"] = match[col].to_numpy()

    set_trace()
    out["PrevGapSec"] = (out["FirstMessage"] - out["PrevDateAnswered"]).dt.total_seconds()
    out["NextGapSec"] = (out["NextDateAnswered"] - out["FirstMessage"]).dt.total_seconds()
    out["NextWithinIntervention"] = out["NextDateAnswered"].le(out["InterventionEnd"])
    out["PrevAndNextSameQuestion"] = out["PrevQuestionId"].eq(out["NextQuestionId"])
    out["PrevAndNextSameSession"] = out["PrevQuizSessionId"].eq(out["NextQuizSessionId"])

    return out



def choose_question(row: pd.Series) -> pd.Series:
    prev_q = row["PrevQuestionId"]
    next_q = row["NextQuestionId"]
    prev_gap = row["PrevGapSec"]
    next_gap = row["NextGapSec"]
    next_within_intervention = bool(row["NextWithinIntervention"]) if pd.notna(row["NextWithinIntervention"]) else False

    # Strongest signal: the student answers a question during the intervention.
    if pd.notna(next_q) and next_within_intervention:
        return pd.Series([
            next_q,
            "next_answer_during_intervention",
            "high",
        ])

    # Strong previous signal: intervention starts very soon after an answer.
    if pd.notna(prev_q) and prev_gap <= 120:
        return pd.Series([
            prev_q,
            "previous_answer_within_2min",
            "high",
        ])

    # If the next answer comes very soon and is clearly closer, the dialogue likely belongs to that problem.
    if (
        pd.notna(next_q)
        and next_gap <= 300
        and (pd.isna(prev_gap) or next_gap < 0.75 * prev_gap)
    ):
        return pd.Series([
            next_q,
            "next_answer_within_5min_and_closer",
            "medium",
        ])

    # Otherwise prefer the recently answered previous question.
    if pd.notna(prev_q) and prev_gap <= 600:
        return pd.Series([
            prev_q,
            "previous_answer_within_10min",
            "medium",
        ])

    # If both sides point to the same question, keep it even with weaker timing.
    if pd.notna(prev_q) and pd.notna(next_q) and prev_q == next_q:
        return pd.Series([
            prev_q,
            "same_question_before_and_after",
            "medium",
        ])

    # Low-confidence fallbacks.
    if pd.notna(prev_q) and prev_gap <= 3600:
        return pd.Series([
            prev_q,
            "previous_answer_within_1h",
            "low",
        ])

    if pd.notna(next_q) and next_gap <= 1800:
        return pd.Series([
            next_q,
            "next_answer_within_30min",
            "low",
        ])

    return pd.Series([pd.NA, "no_confident_match", "none"])



def build_mapping(df_interv: pd.DataFrame, df_usage: pd.DataFrame) -> pd.DataFrame:
    df = get_prev_next_matches(df_interv, df_usage)
    chosen = df.apply(choose_question, axis=1)
    chosen.columns = ["QuestionId", "MatchRule", "MatchConfidence"]
    df = pd.concat([df, chosen], axis=1)
    df["QuestionId"] = df["QuestionId"].astype("Int64")

    # Flag cases worth manual review.
    df["NeedsReview"] = (
        df["QuestionId"].isna()
        | (df["MatchConfidence"] == "low")
        | (
            df["PrevQuestionId"].notna()
            & df["NextQuestionId"].notna()
            & (df["PrevQuestionId"] != df["NextQuestionId"])
            & (df["PrevGapSec"] <= 600)
            & (df["NextGapSec"] <= 600)
        )
    )

    return df



def maybe_augment_dialogues(dialogues_path: str | None, df_mapping: pd.DataFrame, dialogues_out_path: str | None):
    if not dialogues_path:
        return None

    dialogues_file = Path(dialogues_path)
    if not dialogues_file.exists():
        print(f"[WARN] Dialogues file not found: {dialogues_file}. Skipping dialogue augmentation.")
        return None

    if dialogues_out_path is None:
        dialogues_out_path = str(dialogues_file.with_name(dialogues_file.stem + "-with-questions.csv"))

    df_dialogues = pd.read_csv(dialogues_file)
    merge_cols = [
        "UserId",
        "InterventionId",
        "QuestionId",
        "MatchRule",
        "MatchConfidence",
        "NeedsReview",
        "PrevQuestionId",
        "PrevDateAnswered",
        "PrevGapSec",
        "NextQuestionId",
        "NextDateAnswered",
        "NextGapSec",
        "NextWithinIntervention",
    ]
    df_dialogues_q = df_dialogues.merge(
        df_mapping[merge_cols],
        on=["UserId", "InterventionId"],
        how="left",
    )
    df_dialogues_q.to_csv(dialogues_out_path, index=False)
    print(f"[OK] Wrote augmented dialogues: {dialogues_out_path}")
    return dialogues_out_path



def main():
    check = pd.read_csv("data/annotated/eedi/eedi-interventions-with-questions-v2.csv")
    set_trace()


    parser = argparse.ArgumentParser(
        description="Infer the most likely QuestionId for each intervention using both previous and next answers."
    )
    parser.add_argument("--interventions", default="eedi-interventions.csv")
    parser.add_argument("--usage", default="eedi-usage.csv")
    parser.add_argument("--dialogues", default="anonymized-dialogues.csv")
    parser.add_argument("--mapping-out", default="eedi-interventions-with-questions-v2.csv")
    parser.add_argument("--dialogues-out", default="anonymized-dialogues-with-questions-sanity-check.csv")
    args = parser.parse_args()

    df_interv, df_usage = load_inputs(args.interventions, args.usage)
    df_mapping = build_mapping(df_interv, df_usage)
    df_mapping.to_csv(args.mapping_out, index=False)

    print(f"[OK] Wrote intervention mapping: {args.mapping_out}")
    print("\nMatch rule counts:")
    print(df_mapping["MatchRule"].value_counts(dropna=False).to_string())
    print("\nConfidence counts:")
    print(df_mapping["MatchConfidence"].value_counts(dropna=False).to_string())
    print(f"\nNeedsReview rate: {df_mapping['NeedsReview'].mean():.2%}")
    matched_rate = df_mapping["QuestionId"].notna().mean()
    print(f"Matched QuestionId rate: {matched_rate:.2%}")

    maybe_augment_dialogues(args.dialogues, df_mapping, args.dialogues_out)


if __name__ == "__main__":
    main()
