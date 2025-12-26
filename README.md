# Japanese Pitch Accent Dictionary (SQLite)

The most developer-friendly Japanese pitch accent database, optimized for Python and SudachiPy integration.

**124,000+ entries** with pitch patterns, readings, and word origin (goshu) classification.

## Quick Start

```python
import sqlite3

conn = sqlite3.connect("pitch.db")
conn.row_factory = sqlite3.Row

# Lookup pitch accent for 東京
cursor = conn.execute(
    "SELECT * FROM pitch_accents WHERE surface = ? AND reading = ?",
    ("東京", "とうきょう")
)
row = cursor.fetchone()
print(f"Pitch type: {row['accent_pattern']}")  # 0 (heiban)
print(f"Origin: {row['goshu']}")  # proper
```

## Database Schema

```sql
CREATE TABLE pitch_accents (
    id INTEGER PRIMARY KEY,
    surface TEXT NOT NULL,      -- Kanji/kana form (東京)
    reading TEXT NOT NULL,      -- Hiragana reading (とうきょう)
    accent_pattern TEXT,        -- Pitch type: 0=heiban, 1=atamadaka, 2+=nakadaka
    goshu TEXT,                 -- Word origin: wago, kango, gairaigo, proper
    goshu_jp TEXT,              -- Japanese label: 和語, 漢語, 外来語, 固有名詞
    UNIQUE(surface, reading)
);

CREATE INDEX idx_surface ON pitch_accents(surface);
CREATE INDEX idx_reading ON pitch_accents(reading);
```

## Pitch Accent Types

| Type | Name | Pattern | Example |
|------|------|---------|---------|
| 0 | Heiban (平板) | L-H-H-H... | 東京 (とうきょう) |
| 1 | Atamadaka (頭高) | H-L-L-L... | 箸 (はし) |
| 2+ | Nakadaka (中高) | L-H-...-L | 日本 (にほん) |

## Word Origin (Goshu)

| goshu | goshu_jp | Description |
|-------|----------|-------------|
| wago | 和語 | Native Japanese words |
| kango | 漢語 | Sino-Japanese (Chinese origin) |
| gairaigo | 外来語 | Foreign loanwords |
| proper | 固有名詞 | Proper nouns |
| mixed | 混種語 | Hybrid words |

## Data Sources

- **Pitch Accents**: [Kanjium](https://github.com/mifunetoshiro/kanjium) (CC BY-SA 4.0)
- **Goshu Classification**: UniDic + heuristics

## See It In Action

This dictionary powers the analysis engine of **[MieruTone](https://mierutone.com)** — a Japanese pitch accent trainer with:

- Real-time pitch visualization
- Azure Neural TTS audio
- SRS-based practice system

## Contributing

Found an incorrect pitch accent? Help improve the data:

1. **Report an error**: [Open an Issue](../../issues/new?template=pitch-error.yml)
2. **Submit a fix**: [Pull Request Guide](CONTRIBUTING.md)

Your contributions help thousands of Japanese learners get accurate pitch information.

## Integration with SudachiPy

This database is designed to work seamlessly with [SudachiPy](https://github.com/WorksApplications/SudachiPy) Mode C (compound word preservation):

```python
from sudachipy import dictionary, tokenizer
import jaconv
import sqlite3

tok = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C
conn = sqlite3.connect("pitch.db")

def get_pitch(text: str):
    for token in tok.tokenize(text, mode):
        surface = token.surface()
        reading = jaconv.kata2hira(token.reading_form())

        cursor = conn.execute(
            "SELECT accent_pattern FROM pitch_accents WHERE surface = ? AND reading = ?",
            (surface, reading)
        )
        row = cursor.fetchone()
        yield surface, reading, row[0] if row else None

for surface, reading, pitch in get_pitch("東京に行く"):
    print(f"{surface} ({reading}): type {pitch}")
```

## License

Database: **CC BY-SA 4.0** (same as Kanjium source)

If you use this in your project, attribution is appreciated:
```
Pitch accent data from mierutone-dictionary
https://github.com/[your-username]/mierutone-dictionary
```

## Download

- **Latest release**: [pitch.db](../../releases/latest/download/pitch.db) (SQLite, ~15MB)
- **CSV export**: [pitch_accents.csv](../../releases/latest/download/pitch_accents.csv)
