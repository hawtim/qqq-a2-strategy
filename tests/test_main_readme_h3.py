"""Guard main README navigation and frozen H3/A-B source locations.

Does not require licensed market history, does not test strategy returns,
and cannot authorize trading.
"""
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[1]

class MainReadmeH3Tests(unittest.TestCase):
    def test_h3_and_ab_are_frontpage_first(self):
        text=(ROOT/"README.md").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# H3 趋势量化策略"))
        self.assertIn("docs/STRATEGY_H3_FROZEN.md",text)
        self.assertIn("docs/STRATEGY_H3_PROFIT_GUARD_AB_V1.md",text)
        self.assertIn("research/config/h3_profit_guard_ab_v1.json",text)
        self.assertIn("docs/research/INDEX.md",text)
        self.assertIn("**H3（基础策略）**",text)
        self.assertIn("**H3 + A v1.0**",text)
        self.assertIn("**H3 + B v1.0**",text)
        self.assertLess(len(text),12000,"Research logs belong in docs/research/INDEX.md")
        self.assertNotIn("## 当前选定研究基准 V3",text)
    def test_local_markdown_links_have_targets(self):
        for src in (
            ROOT/"README.md",
            ROOT/"docs/STRATEGY_H3_FROZEN.md",
            ROOT/"docs/research/INDEX.md",
        ):
            doc=src.read_text(encoding="utf-8")
            for dest in re.findall(r"\]\(([^)\n]+)\)",doc):
                relative=dest.split("#",1)[0]
                if not relative or relative.startswith(("http:","https:","mailto:","#")):
                    continue
                with self.subTest(source=src.name, link=dest):
                    self.assertTrue((src.parent/relative).exists(),
                        "Broken local Markdown link")
    def test_frozen_h3_ab_paths_are_distinct(self):
        h3=ROOT/"research/experiments/a2_v3_1_h3_trend_spread_quick_reentry.py"
        ab=ROOT/"research/experiments/h3_profit_guard_ab_v1_frozen.py"
        self.assertTrue(h3.is_file())
        self.assertTrue(ab.is_file())
        self.assertTrue((ROOT/"research/config/h3_profit_guard_ab_v1.json").is_file())

if __name__=="__main__":
    unittest.main()
