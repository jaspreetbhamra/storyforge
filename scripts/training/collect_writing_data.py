"""
StoryForge Data Collection Script
Collects creative writing data from multiple sources
"""

import json
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


class CreativeWritingCollector:
    """Collects and preprocesses creative writing datasets"""

    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_writingprompts(self, max_samples=50000):
        """
        Collect from WritingPrompts dataset
        Reddit r/WritingPrompts - high quality creative shorts
        """
        print("📚 Collecting WritingPrompts dataset...")

        try:
            # Load from HuggingFace
            dataset = load_dataset("euclaise/writingprompts", split="train")

            # Filter and process
            stories = []
            for item in tqdm(dataset, desc="Processing WritingPrompts"):
                if len(stories) >= max_samples:
                    break

                # Extract prompt and story
                prompt = item.get("prompt", "").strip()
                story = item.get("story", "").strip()

                # Quality filters
                if (
                    len(story) < 200
                    or len(story) > 8000
                    or len(prompt) < 10
                    or len(prompt) > 500
                ):
                    continue

                stories.append(
                    {
                        "prompt": prompt,
                        "text": story,
                        "source": "writingprompts",
                        "length": len(story),
                        "genre": "mixed",
                    }
                )

            # Save
            output_file = self.output_dir / "writingprompts.jsonl"
            self._save_jsonl(stories, output_file)
            print(f"✅ Saved {len(stories)} WritingPrompts stories")
            return len(stories)

        except Exception as e:
            print(f"❌ Error collecting WritingPrompts: {e}")
            return 0

    def collect_tiny_stories(self, max_samples=20000):
        """
        Collect TinyStories - synthetically generated children's stories
        Good for structure and coherence
        """
        print("📚 Collecting TinyStories dataset...")

        try:
            dataset = load_dataset("roneneldan/TinyStories", split="train")

            stories = []
            for item in tqdm(dataset, desc="Processing TinyStories"):
                if len(stories) >= max_samples:
                    break

                text = item.get("text", "").strip()

                # Quality filters
                if len(text) < 100 or len(text) > 2000:
                    continue

                stories.append(
                    {
                        "text": text,
                        "source": "tinystories",
                        "length": len(text),
                        "genre": "children",
                    }
                )

            output_file = self.output_dir / "tinystories.jsonl"
            self._save_jsonl(stories, output_file)
            print(f"✅ Saved {len(stories)} TinyStories")
            return len(stories)

        except Exception as e:
            print(f"❌ Error collecting TinyStories: {e}")
            return 0

    def collect_bookcorpus_samples(self, max_samples=10000):
        """
        Collect samples from books dataset
        Note: This is a smaller sample as full BookCorpus is large
        """
        print("📚 Collecting BookCorpus samples...")

        try:
            # Using the public domain books subset
            dataset = load_dataset("bookcorpus", split="train", streaming=True)

            stories = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break

                text = item.get("text", "").strip()

                # We want longer coherent passages
                if len(text) < 500 or len(text) > 10000:
                    continue

                stories.append(
                    {
                        "text": text,
                        "source": "bookcorpus",
                        "length": len(text),
                        "genre": "novel",
                    }
                )

                if len(stories) % 1000 == 0:
                    print(f"  Collected {len(stories)} book passages...")

            output_file = self.output_dir / "bookcorpus.jsonl"
            self._save_jsonl(stories, output_file)
            print(f"✅ Saved {len(stories)} BookCorpus passages")
            return len(stories)

        except Exception as e:
            print(f"❌ Error collecting BookCorpus: {e}")
            print("  Note: BookCorpus may require special access")
            return 0

    def collect_all(self):
        """Collect from all sources"""
        print("\n" + "=" * 60)
        print("🎯 StoryForge Data Collection")
        print("=" * 60 + "\n")

        total_collected = 0

        # Collect from each source
        total_collected += self.collect_writingprompts(max_samples=50000)
        total_collected += self.collect_tiny_stories(max_samples=20000)
        # total_collected += self.collect_bookcorpus_samples(max_samples=10000)  # Optional

        print("\n" + "=" * 60)
        print(f"✅ Total stories collected: {total_collected:,}")
        print("=" * 60 + "\n")

        return total_collected

    def _save_jsonl(self, data, filepath):
        """Save data in JSONL format"""
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def create_analysis_report(self):
        """Create a summary report of collected data"""
        print("\n📊 Generating Data Analysis Report...")

        all_files = list(self.output_dir.glob("*.jsonl"))

        if not all_files:
            print("❌ No data files found!")
            return

        report = []
        total_stories = 0

        for file in all_files:
            print(f"\n  Analyzing {file.name}...")

            # Load data
            stories = []
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    stories.append(json.loads(line))

            # Calculate statistics
            lengths = [s["length"] for s in stories]
            avg_length = sum(lengths) / len(lengths) if lengths else 0
            min_length = min(lengths) if lengths else 0
            max_length = max(lengths) if lengths else 0

            report.append(
                {
                    "source": file.stem,
                    "count": len(stories),
                    "avg_length": int(avg_length),
                    "min_length": min_length,
                    "max_length": max_length,
                }
            )

            total_stories += len(stories)

        # Print report
        print("\n" + "=" * 60)
        print("📊 DATA COLLECTION SUMMARY")
        print("=" * 60)

        df = pd.DataFrame(report)
        print(df.to_string(index=False))

        print(f"\n📈 Total Stories: {total_stories:,}")
        print(f"💾 Estimated Size: ~{total_stories * 2 / 1000:.1f} MB")
        print("=" * 60 + "\n")

        # Save report
        report_file = self.output_dir / "collection_report.json"
        with open(report_file, "w") as f:
            json.dump(
                {
                    "total_stories": total_stories,
                    "sources": report,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                indent=2,
            )

        print(f"💾 Report saved to: {report_file}")


def main():
    """Main execution"""
    print(
        """
    ╔════════════════════════════════════════════╗
    ║     StoryForge Data Collection System     ║
    ║                                            ║
    ║  Collecting creative writing datasets      ║
    ║  for fine-tuning Llama 3.1 8B             ║
    ╚════════════════════════════════════════════╝
    """
    )

    # Initialize collector
    collector = CreativeWritingCollector(output_dir="data/raw")

    # Collect data
    print("🚀 Starting data collection...")
    print("⏱️  This may take 10-20 minutes depending on your internet speed\n")

    total = collector.collect_all()

    if total > 0:
        # Generate report
        collector.create_analysis_report()

        print("\n✅ Data collection complete!")
        print("\n📋 Next Steps:")
        print("   1. Review the data in data/raw/")
        print("   2. Run data preprocessing script")
        print("   3. Start fine-tuning!")
        print("\n💡 Tip: Check collection_report.json for statistics")
    else:
        print("\n❌ No data collected. Please check your internet connection.")
        print("   You may also need to authenticate with HuggingFace:")
        print("   Run: huggingface-cli login")


if __name__ == "__main__":
    main()
