"""
main.py — End-to-End Emotion Detection Pipeline Runner
========================================================
Orchestrates the full workflow:
    1. Build / load dataset (synthetic or CSV)
    2. Preprocess all text
    3. Train & evaluate models
    4. Run interactive inference demo

Run from the project root:
    python main.py

Or with a custom CSV dataset:
    python main.py --data path/to/dataset.csv --text-col text --label-col emotion
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from src.preprocess import preprocess_dataframe
from src.train import train_and_evaluate
from src.predict import EmotionPredictor


# ---------------------------------------------------------------------------
# Windows Console Unicode Safety
# ---------------------------------------------------------------------------
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Synthetic Dataset
# ---------------------------------------------------------------------------
def build_synthetic_dataset() -> pd.DataFrame:
    """
    Create a small but representative mock dataset so the pipeline is
    instantly runnable without any external downloads.

    The dataset covers 6 emotion classes with ~15-20 samples each.
    When you're ready, replace this with a real dataset such as:
      - Hugging Face: dair-ai/emotion
      - Kaggle: Emotion Dataset for NLP
      - Twitter Sentiment140
    """
    data = {
        "text": [
            # ---- HAPPY ----
            "I just got promoted at work! Best day ever!",
            "Sunshine and smiles, today is absolutely wonderful!",
            "My best friend surprised me with a birthday cake 🎂",
            "I passed all my exams with flying colors!",
            "Finally got tickets to see my favorite band live!",
            "Today was the most amazing day of my entire life",
            "I'm so grateful for all the love and support around me",
            "We won the championship game! Unbelievable feeling!",
            "Got accepted into my dream university! Can't stop smiling!",
            "The weather is perfect and everything feels right today",
            "My dog learned a new trick and I'm so proud of him",
            "Had the best dinner date with my partner tonight",
            "I finished my project ahead of deadline, feeling accomplished!",
            "Just received the most thoughtful gift from my colleague",
            "Life is beautiful when you have wonderful people around you",
            "I'm absolutely thrilled about my new job offer!",
            "I am so thrilled and ecstatic about the great news!",
            "This is truly wonderful and makes me incredibly happy!",
            "i am happy today",
            "I feel so happy and full of joy!",
            "This makes me extremely happy!",
            "Happy days are here again!",

            # ---- SAD ----
            "I feel so lonely tonight, nobody seems to care anymore",
            "My heart is broken and I don't know how to move on",
            "Lost my grandmother last week. The grief is overwhelming.",
            "I failed the interview again. Starting to lose hope.",
            "Watching everyone move forward while I'm stuck is painful",
            "The rain matches my mood perfectly today... just grey",
            "I miss my childhood friends, we've all drifted apart",
            "Feeling empty inside, like nothing really matters anymore",
            "Another sleepless night filled with tears and regret",
            "I wish things could go back to the way they were before",
            "My pet passed away and I feel completely devastated",
            "Nobody came to my birthday party. I feel invisible.",
            "The loneliness is crushing me, I need someone to talk to",
            "I got rejected from every job I applied for this month",
            "Sometimes I wonder if anyone truly understands how I feel",
            "I'm crying because I miss her so much it hurts",
            "Tears won't stop falling, I miss you more than words can say",
            "The pain of missing someone hurts so deeply in my chest",
            "i am sad",
            "I feel very sad and depressed today.",
            "This is such a sad situation.",
            "I am incredibly sad about what happened.",

            # ---- ANGRY ----
            "I can't believe they lied to me again! So furious right now!",
            "This is absolutely unacceptable! I demand an explanation!",
            "The customer service was the worst I've ever experienced!",
            "How dare they take credit for MY work! I'm livid!",
            "Stop wasting my time with these ridiculous excuses!",
            "I'm sick and tired of being treated like I don't matter!",
            "The corruption in this system makes my blood boil!",
            "Every single day there's a new problem. I've had enough!",
            "They cancelled my flight without any notice or compensation!",
            "I hate when people don't respect boundaries. So disrespectful!",
            "This traffic jam is driving me absolutely insane right now",
            "My neighbor's loud music at 3 AM is beyond inconsiderate",
            "The referee made the worst call I have ever seen in my life",
            "People who litter make me absolutely furious beyond words",
            "I can't stand the hypocrisy anymore, it's infuriating!",
            "This incompetent service is making me lose my mind!",
            "The sheer incompetence of this staff is completely infuriating!",
            "I am about to lose my mind over this terrible customer service!",
            "i hate you",
            "fuck you",
            "I absolutely hate this!",
            "This makes me so angry, I hate it!",

            # ---- SURPRISE ----
            "Wait, WHAT?! I never expected that plot twist!",
            "Oh my god, I can't believe you're here! What a shock!",
            "I just found out I won the lottery! Is this real life?",
            "No way! They're actually getting married? Since when?!",
            "I opened the door and everyone yelled surprise! I nearly fainted!",
            "The test results came back and I'm completely stunned",
            "I never in a million years thought this would happen to me",
            "Plot twist of the century! My jaw literally dropped!",
            "They showed up unannounced after being away for five years!",
            "I can't process what just happened, this is mind-blowing!",
            "Whoa, did that really just happen? I'm in total disbelief!",
            "Found a hidden room behind my bookshelf! Unbelievable!",
            "The magician's trick left the entire audience speechless",
            "I accidentally discovered my neighbor is a famous author",
            "The ending of that movie completely caught me off guard!",
            "Wow, I completely didn't expect that incredible twist!",
            "My jaw hit the floor when I heard the shocking news!",
            "I am totally speechless, what an amazing surprise!",
            "What a surprise!",
            "I am surprised by this outcome!",
            "This is surprisingly good!",
            "I didn't see that coming at all!",

            # ---- FEAR ----
            "I heard strange noises outside my window and I'm terrified",
            "The thought of public speaking makes me want to run away",
            "I'm scared something terrible is going to happen tomorrow",
            "Walking alone at night in this neighborhood frightens me",
            "The horror movie gave me nightmares for three days straight",
            "I'm terrified of losing everything I've worked so hard for",
            "The turbulence on that flight was absolutely horrifying",
            "I panicked when I realized I was completely lost and alone",
            "There's a spider on my wall and I'm frozen with fear",
            "The deadline is tomorrow and I haven't even started yet",
            "I'm anxious about the medical test results coming next week",
            "The earthquake shook the building and I was paralyzed",
            "Being stuck in the elevator triggered my claustrophobia",
            "I dread going back to that place after what happened",
            "The darkness in the basement gives me the creeps every time",
            "I am trembling with fear, this is so terrifying",
            "The creepy sounds in the dark make me so nervous and scared",
            "I'm extremely anxious and terrified of what might happen next",
            "what if i fail in my exam",
            "I am scared that I will fail my exam",
            "The fear of failure is paralyzing me",
            "I'm terrified of failing this test",

            # ---- NEUTRAL ----
            "The meeting is scheduled for 3 PM in the conference room",
            "I went to the grocery store and bought some vegetables",
            "The report contains data from the last fiscal quarter",
            "Please send me the updated spreadsheet by end of day",
            "The temperature outside is around 72 degrees Fahrenheit",
            "I took the bus to work today instead of driving my car",
            "The new software update will be released next Tuesday",
            "I had cereal for breakfast and a sandwich for lunch today",
            "The library closes at 9 PM on weekdays and 5 PM weekends",
            "My commute takes about forty-five minutes each morning",
            "The document needs to be reviewed before final submission",
            "I attended the weekly standup meeting this morning as usual",
            "The package should arrive within three to five business days",
            "She mentioned the project timeline during yesterday's call",
            "The printer on the second floor is currently out of paper",
            "The system update will be completed by tomorrow morning.",
            "Please verify the details in the attached spreadsheet.",
            "The temperature is expected to remain stable throughout the week.",
            "I am reading a book.",
            "The wall is painted white.",
            "A chair has four legs.",
            "Water boils at 100 degrees Celsius.",
        ],
        "emotion": (
            ["happy"] * 22
            + ["sad"] * 22
            + ["angry"] * 22
            + ["surprise"] * 22
            + ["fear"] * 22
            + ["neutral"] * 22
        ),
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Emotion Detection from Text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                       # Run with synthetic data
  python main.py --data emotion.csv                    # Use a custom CSV file
  python main.py --data emotion.csv --text-col tweet   # Custom column names
        """,
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to a CSV file with text and label columns.",
    )
    parser.add_argument(
        "--text-col", type=str, default="text",
        help="Name of the text column in the CSV (default: 'text').",
    )
    parser.add_argument(
        "--label-col", type=str, default="emotion",
        help="Name of the label column in the CSV (default: 'emotion').",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directory to save models, plots, and reports (default: 'outputs').",
    )
    parser.add_argument(
        "--max-features", type=int, default=10_000,
        help="Max vocabulary size for TF-IDF (default: 10000).",
    )
    parser.add_argument(
        "--no-cross-val", action="store_true",
        help="Skip 5-fold cross-validation to speed up training.",
    )
    args = parser.parse_args()

    # ======================================================================
    # STEP 1: Load Data
    # ======================================================================
    print("\n" + "=" * 65)
    print("  STEP 1 / 4 — DATA LOADING")
    print("=" * 65)

    if args.data:
        print(f"  Loading dataset from: {args.data}")
        df = pd.read_csv(args.data)
        # Rename columns to internal names if needed
        if args.text_col != "text":
            df = df.rename(columns={args.text_col: "text"})
        if args.label_col != "emotion":
            df = df.rename(columns={args.label_col: "emotion"})
    else:
        print("  No --data flag provided. Using built-in synthetic dataset.")
        print("  TIP: Use --data <path.csv> to plug in your own dataset.")
        df = build_synthetic_dataset()

    print(f"  Loaded {len(df):,} samples across {df['emotion'].nunique()} classes.")
    print(f"  Class distribution:\n{df['emotion'].value_counts().to_string()}")

    # ======================================================================
    # STEP 2: Preprocess Text
    # ======================================================================
    print("\n" + "=" * 65)
    print("  STEP 2 / 4 — TEXT PREPROCESSING")
    print("=" * 65)

    df = preprocess_dataframe(df, text_column="text", output_column="clean_text")

    # Show a few examples
    print("\n  Sample preprocessed texts:")
    for i, row in df.head(3).iterrows():
        print(f"    RAW  : {row['text'][:80]}...")
        print(f"    CLEAN: {row['clean_text'][:80]}")
        print()

    # ======================================================================
    # STEP 3: Train & Evaluate Models
    # ======================================================================
    print("=" * 65)
    print("  STEP 3 / 4 — MODEL TRAINING & EVALUATION")
    print("=" * 65)

    best_model, vectorizer, results = train_and_evaluate(
        df,
        text_column="clean_text",
        label_column="emotion",
        output_dir=args.output_dir,
        max_features=args.max_features,
        run_cross_val=not args.no_cross_val,
    )

    # ======================================================================
    # STEP 4: Interactive Inference Demo
    # ======================================================================
    print("\n" + "=" * 65)
    print("  STEP 4 / 4 — INFERENCE DEMO")
    print("=" * 65)

    predictor = EmotionPredictor(
        model=best_model,
        vectorizer=vectorizer,
        class_names=sorted(df["emotion"].unique().tolist()),
        model_name="BestModel",
    )

    demo_sentences = [
        "I'm absolutely thrilled about my new job offer!",
        "Why does nobody ever listen to what I have to say?",
        "I can't believe they threw me a surprise party!",
        "The darkness in the alley made me extremely nervous",
        "Please forward the meeting notes to the team by Friday",
        "I'm crying because I miss her so much it hurts",
        "This incompetent service is making me lose my mind!",
    ]

    print("\n  Running predictions on demo sentences:\n")
    for sentence in demo_sentences:
        result = predictor.predict(sentence)
        prob_bar = "█" * int(result["confidence"] * 30)
        print(f"  📝 \"{sentence[:65]}...\"")
        print(f"     → {result['label'].upper():10s}  "
              f"({result['confidence']:.1%})  {prob_bar}")
        print()

    # ---- Interactive mode ----
    print("─" * 65)
    print("  INTERACTIVE MODE — Type a sentence to classify.")
    print("  Type 'quit' or 'exit' to stop.")
    print("─" * 65)

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("  Goodbye! 👋")
            break

        if not user_input:
            continue

        result = predictor.predict(user_input)
        print(f"  → Emotion   : {result['label'].upper()}")
        print(f"  → Confidence: {result['confidence']:.1%}")
        print(f"  → All scores: ", end="")
        sorted_probs = sorted(
            result["probabilities"].items(), key=lambda x: x[1], reverse=True
        )
        print(" | ".join(f"{k}: {v:.1%}" for k, v in sorted_probs))


if __name__ == "__main__":
    main()
