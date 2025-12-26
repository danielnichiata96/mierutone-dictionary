# Contributing to mierutone-dictionary

Thank you for helping improve Japanese pitch accent data! Your contributions help thousands of learners.

## Reporting a Pitch Accent Error

The easiest way to contribute:

1. Go to [Issues](../../issues/new?template=pitch-error.yml)
2. Fill in the word, current (wrong) pitch, and correct pitch
3. Include a source if possible (NHK dictionary, native speaker, etc.)

## Submitting a Fix via Pull Request

### 1. Fork and clone

```bash
git clone https://github.com/YOUR_USERNAME/mierutone-dictionary.git
cd mierutone-dictionary
```

### 2. Edit the CSV

The source of truth is `data/corrections.csv`. Add your correction:

```csv
surface,reading,accent_pattern,goshu,goshu_jp,source,notes
橋,はし,2,wago,,NHK,was incorrectly 1
```

> **Note**: `goshu_jp` is optional — it's auto-derived from `goshu` if omitted.
> Multiple pitch patterns can be specified as `0,2`.

### 3. Validate your change

```bash
python scripts/validate.py
```

### 4. Submit PR

```bash
git checkout -b fix/hashi-pitch
git add data/corrections.csv
git commit -m "fix: 橋 pitch accent 1→2"
git push origin fix/hashi-pitch
```

Then open a Pull Request.

## Correction Priority

We prioritize fixes for:

1. **Common words** (JLPT N5-N3 vocabulary)
2. **Homophones** (橋/箸/端 - where pitch distinguishes meaning)
3. **Verified sources** (NHK Accent Dictionary, OJAD, native speakers)

## What Makes a Good Source?

| Source | Reliability |
|--------|-------------|
| NHK日本語発音アクセント新辞典 | Highest |
| OJAD (Online Japanese Accent Dictionary) | High |
| Native speaker recording | High (with context) |
| Other pitch dictionaries | Medium |
| "I heard it this way" | Low (needs verification) |

## Goshu Classification

If you're also correcting word origin (goshu):

| goshu | When to use |
|-------|-------------|
| wago | Native Japanese (やま, うみ, たべる) |
| kango | Chinese-origin (山岳, 海洋, 食事) |
| gairaigo | Western loanwords (パン, コンピュータ) |
| proper | Names, places (東京, 田中) |
| mixed | Hybrid (消しゴム = 消し + ゴム) |

## Code of Conduct

- Be respectful in discussions
- Provide sources when possible
- Accept that regional variations exist (Tokyo vs Osaka vs Kyushu)
- For disputed accents, we default to Tokyo standard (標準語)

## Questions?

Open a [Discussion](../../discussions) or check [MieruTone](https://mierutone.com) for context on how this data is used.
