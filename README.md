# Table Transformer (TATR) — Detection & OCR Parsing

Two notebooks demonstrating Microsoft's [Table Transformer](https://huggingface.co/microsoft/table-transformer-detection) for end-to-end table extraction from document images.

---

## Notebooks

### 1. `table_detection.ipynb` — Table Detection & Structure Recognition

Uses `microsoft/table-transformer-detection` (DETR-based) to:
- **Detect** table regions in document images
- **Recognize** table structure (rows, columns, headers)

DETR (DEtection TRansformer) combines a CNN backbone with a Transformer encoder-decoder for object detection without anchor boxes.

### 2. `table_parsing_ocr.ipynb` — Full Table Parsing with EasyOCR

End-to-end pipeline:
1. Detect tables with TATR
2. Recognize cell structure
3. Run **EasyOCR** on each cell
4. Reconstruct a structured table (rows × columns)

```python
from transformers import TableTransformerForObjectDetection
model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection"
)
```

---

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline Overview

```
Document Image
     │
     ▼
Table Detection (TATR)
     │
     ▼
Structure Recognition (TATR)
     │
     ▼
Cell Bounding Boxes
     │
     ▼
EasyOCR → Text per cell
     │
     ▼
Structured Table (DataFrame)
```

## References

- [Table Transformer Paper](https://arxiv.org/abs/2110.00061)
- [HuggingFace Model Hub](https://huggingface.co/microsoft/table-transformer-detection)

## License

MIT
