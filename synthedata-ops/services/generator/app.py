import os
import argparse
import pandas as pd
from ctgan import CTGAN

def run_generation(data_path, output_path, num_samples, epochs):
    print(f"🔄 Loading source dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Drop identifying features if they exist in the survey
    for col in ['Name', 'Address']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Categorize features dynamically based on survey responses
    categorical_features = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 12:
            categorical_features.append(col)

    print(f"📊 Identified Categorical Features: {categorical_features}")

    print(f"🤖 Training Tabular GAN for {epochs} epochs...")
    model = CTGAN(epochs=epochs)
    model.fit(df, categorical_features)

    print(f"✨ Generating {num_samples} privacy-preserving records...")
    synthetic_df = model.sample(num_samples)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    synthetic_df.to_csv(output_path, index=False)
    print(f"✅ Synthetic pool saved successfully to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    run_generation(args.data_path, args.output_path, args.samples, args.epochs)