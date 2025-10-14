"""
Data Preprocessing for StoryForge
Cleans, formats, and prepares data for fine-tuning
"""

import json
import random
import re
from pathlib import Path

from tqdm import tqdm


class StoryPreprocessor:
    """Preprocesses creative writing data for fine-tuning"""

    def __init__(self, input_dir="data/raw", output_dir="data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Quality thresholds
        self.min_length = 200
        self.max_length = 8000
        self.min_words = 50

    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove multiple newlines (keep max 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Fix common issues
        text = text.replace("\r", "\n")
        text = re.sub(r"\n ", "\n", text)  # Remove space after newline

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Remove markdown artifacts common in Reddit posts
        text = re.sub(r"\[WP\]|\[PI\]|\[TT\]", "", text, flags=re.IGNORECASE)

        # Strip and clean
        text = text.strip()

        return text

    def is_quality_story(self, text):
        """Check if story meets quality criteria"""
        # Length checks
        if len(text) < self.min_length or len(text) > self.max_length:
            return False

        # Word count check
        word_count = len(text.split())
        if word_count < self.min_words:
            return False

        # Check for too many special characters (likely corrupted)
        special_char_ratio = len(re.findall(r"[^a-zA-Z0-9\s\.,!?\'\"-]", text)) / len(
            text
        )
        if special_char_ratio > 0.1:
            return False

        # Check for repeated patterns (spam/corrupted)
        if self._has_repeated_patterns(text):
            return False

        # Check for minimum dialogue/narrative structure
        # Good stories usually have some punctuation variety
        has_periods = "." in text
        # has_dialogue = '"' in text or "'" in text

        if not has_periods:
            return False

        return True

    def _has_repeated_patterns(self, text, threshold=0.3):
        """Detect if text has too many repeated patterns"""
        # Check for repeated sentences
        sentences = text.split(".")
        if len(sentences) < 2:
            return False

        unique_sentences = len(set(s.strip() for s in sentences if s.strip()))
        if unique_sentences / len(sentences) < threshold:
            return True

        return False

    def format_for_training(self, story, include_prompt=True):
        """
        Format story for instruction fine-tuning
        Llama 3.1 instruction format
        """
        if include_prompt and "prompt" in story:
            # Instruction-following format
            formatted = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a creative writer. Write engaging, well-structured stories based on the given prompts.<|eot_id|><|start_header_id|>user<|end_header_id|>

{story['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{story['text']}<|eot_id|>"""
        else:
            # Pure continuation format (for stories without prompts)
            formatted = f"""<|begin_of_text|>{story['text']}<|eot_id|>"""

        return formatted

    def process_dataset(self, input_file):
        """Process a single dataset file"""
        print(f"\n📝 Processing {input_file.name}...")

        # Load data
        stories = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                stories.append(json.loads(line))

        print(f"  Loaded {len(stories)} stories")

        # Clean and filter
        processed_stories = []

        for story in tqdm(stories, desc="  Cleaning & filtering"):
            # Clean text
            cleaned_text = self.clean_text(story["text"])

            # Clean prompt if exists
            if "prompt" in story:
                story["prompt"] = self.clean_text(story["prompt"])

            # Quality check
            if not self.is_quality_story(cleaned_text):
                continue

            story["text"] = cleaned_text
            processed_stories.append(story)

        print(
            f"  ✅ Kept {len(processed_stories)}/{len(stories)} stories ({len(processed_stories) / len(stories) * 100:.1f}%)"
        )

        return processed_stories

    def create_train_val_split(self, all_stories, val_ratio=0.05):
        """Split data into train and validation sets"""
        print(
            f"\n🔀 Creating train/validation split ({val_ratio * 100:.0f}% validation)..."
        )

        # Shuffle
        random.shuffle(all_stories)

        # Split
        split_idx = int(len(all_stories) * (1 - val_ratio))
        train_stories = all_stories[:split_idx]
        val_stories = all_stories[split_idx:]

        print(f"  📊 Train: {len(train_stories):,} stories")
        print(f"  📊 Validation: {len(val_stories):,} stories")

        return train_stories, val_stories

    def save_formatted_data(self, stories, output_file, format_type="instruction"):
        """Save formatted data for training"""
        print(f"\n💾 Saving to {output_file}...")

        with open(output_file, "w", encoding="utf-8") as f:
            for story in tqdm(stories, desc="  Formatting"):
                # Format based on whether story has prompt
                include_prompt = "prompt" in story and format_type == "instruction"
                formatted = self.format_for_training(story, include_prompt)

                # Save as JSONL
                f.write(
                    json.dumps(
                        {
                            "text": formatted,
                            "source": story.get("source", "unknown"),
                            "length": len(formatted),
                        }
                    )
                    + "\n"
                )

        print(f"  ✅ Saved {len(stories)} formatted stories")

    def process_all(self):
        """Process all datasets"""
        print("\n" + "=" * 60)
        print("🎯 StoryForge Data Preprocessing")
        print("=" * 60)

        # Find all input files
        input_files = list(self.input_dir.glob("*.jsonl"))

        if not input_files:
            print("❌ No data files found in data/raw/")
            print("   Run collect_writing_data.py first!")
            return

        # Process each file
        all_stories = []
        for input_file in input_files:
            if input_file.name == "collection_report.json":
                continue
            processed = self.process_dataset(input_file)
            all_stories.extend(processed)

        print(f"\n📊 Total processed stories: {len(all_stories):,}")

        if len(all_stories) == 0:
            print("❌ No valid stories after preprocessing!")
            return

        # Create train/val split
        train_stories, val_stories = self.create_train_val_split(all_stories)

        # Save formatted data
        self.save_formatted_data(
            train_stories, self.output_dir / "train.jsonl", format_type="instruction"
        )

        self.save_formatted_data(
            val_stories, self.output_dir / "val.jsonl", format_type="instruction"
        )

        # Create metadata
        metadata = {
            "total_stories": len(all_stories),
            "train_stories": len(train_stories),
            "val_stories": len(val_stories),
            "avg_length": sum(s["length"] for s in all_stories) / len(all_stories),
            "sources": list(set(s.get("source", "unknown") for s in all_stories)),
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 60)
        print("✅ Preprocessing Complete!")
        print("=" * 60)
        print(f"\n📁 Output location: {self.output_dir}")
        print(f"📊 Train set: {len(train_stories):,} stories")
        print(f"📊 Validation set: {len(val_stories):,} stories")
        print("\n📋 Files created:")
        print("   • train.jsonl")
        print("   • val.jsonl")
        print("   • metadata.json")

        print("\n💡 Next step: Start fine-tuning!")
        print("   python scripts/training/finetune_base.py")


def main():
    """Main execution"""
    print(
        """
    ╔════════════════════════════════════════════╗
    ║    StoryForge Data Preprocessing System   ║
    ║                                            ║
    ║  Cleaning and formatting creative writing ║
    ║  data for Llama 3.1 fine-tuning          ║
    ╚════════════════════════════════════════════╝
    """
    )

    # Set random seed for reproducibility
    random.seed(42)

    # Initialize preprocessor
    preprocessor = StoryPreprocessor(input_dir="data/raw", output_dir="data/processed")

    # Process all data
    preprocessor.process_all()


if __name__ == "__main__":
    main()
