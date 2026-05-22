# Style Guide (`book/`)

All Markdown in this book follows this file. When revising, update the style guide first, then individual chapters.

## Chapter Titles

```markdown
# Chapter N · English Title
```

The preface uses `# Preface: …` and is not called "Chapter 0"; the full book scope is **Preface + Chapters 1–7 + Appendices A–D**.

## Chapter Header Metadata (Required for ch01–ch07)

```markdown
> **Estimated reading**: Body XX min | Run code XX min | Deep dive XX min
> **Companion code**: [`chNN_xxx/`](https://github.com/binbinao/physicsnemo-from-zero-to-one/tree/main/chNN_xxx)
> **Difficulty**: ⭐…
> **Key terms**: `…`
> **Environment baseline**: See [ENVIRONMENT.md](../docs/ENVIRONMENT.md) · PhysicsNeMo v2.0 · PyTorch ≥ 2.3
```

Optional: `> **📌 Before you read**` (when the narrative and default code differ, e.g., ch04, ch07).

## Section Titles

| Type | Format |
|:---|:---|
| Hook | `## N.0 Hook: …` |
| Quick Track | `## N.x 🟢 Quick Track: …` |
| Deep Dive | `## N.x 🔵 …` |
| Industry Mapping | `## N.x 🏭 Industry Mapping: …` |
| Failure | `## N.x 🔵 Failure Case: …` |
| Chapter End | `## N.x ➡️ Next Chapter Preview + Chapter-End CTA` |

## Figures and Tables

- **Figure numbering**: `F{chapter}.{seq}`, subfigures `F2.7.1` only when multiple figures in the same section require sub-numbering.
- **Table numbering**: `T{chapter}.{seq}`, title `### T1.1 …`.
- **Figure captions**: `![F1.0 Short description](assets/f1_0_banner.png)`; production-use descriptions go in [assets/FIGURE_MANIFEST.md](assets/FIGURE_MANIFEST.md), not in the body text.
- **Mermaid**: For print and static reading, use `assets/*.png` as the source of truth; **do not** place a `` ```text` `` vertical-flow or Mermaid block alongside the finalized figure showing the same content (triple duplication). For architecture/algorithm flows, see [BOOK_FIGURE_OUTPUT_FORMS.md](../docs/BOOK_FIGURE_OUTPUT_FORMS.md).
- **Architecture diagrams** (FNO block, Domain/Constraint, etc.): Use script-reproducible vector-style PNGs (`book/scripts/generate_diagram_figures.py --vector-only`) or Figma/SVG source files; text inside diagrams in English, captions in the target language.

## Chapter-End Blocks (Fixed Order)

1. `## N.x ➡️ Next Chapter Preview + Chapter-End CTA` (final chapter uses epilogue + appendix guide)
2. Chapter-end CTA blockquote (📘 💬 🔔)
3. Video script note (link to [video_scripts/README.md](video_scripts/README.md))
4. `### Further Reading`
5. Footer: `*Chapter word count: ~X,XXX words · Figures/tables: N · Version: v1.0 · Updated: YYYY-MM-DD*`

## Appendices

- Title: `# Appendix X · Title`
- Sections: `## X.1 …`, numbered consecutively
- Footer: `*Appendix X · v1.0 · Updated: YYYY-MM-DD*`

## Punctuation and Quotation Marks

- Body text uses standard English quotation marks (" ") for emphasis and citations.
- English proper nouns, code, and paths retain their original form.
