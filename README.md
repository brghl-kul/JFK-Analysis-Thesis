1.0 Text Acquisition
Scanned PDF documents were converted into machine-readable text using Optical Character Recognition. Due to inconsistent OCR results caused by degraded scans and formatting artifacts, AI-assisted document transcriptions were used and converted into standardized text files (doctly.ai).
Technique used:
• Optical Character Recognition (OCR) – Tesseract (Smith, 2007)

1.1 Text Cleaning
The text corpus was cleaned to remove archival headers, document identifiers, and formatting artifacts, while normalizing whitespace and punctuation.
Techniques used:
• Text preprocessing
• Text normalization

1.2 Entity and Relation Extraction
Named entities such as persons, organizations, or locations were extracted using Natural Language Processing. Relationships between entities were identified using document-level co-occurrence.
Techniques used:
• Named Entity Recognition (NER)
• Co-occurrence-based relation extraction

1.3 Network Construction
A weighted entity co-occurrence network was created where nodes represent entities and edge weights represent the frequency of co-occurrence.
Techniques used:
• Network construction using NetworkX
• Weighted network analysis

1.4 Community Detection
The Louvain algorithm was applied to detect clusters of closely connected actors within the network.
Technique used:
• Louvain community detection (Blondel et al., 2008)

1.5 Entity Frequency Analysis
Entity frequency analysis was performed to identify the most frequently mentioned actors in the corpus.
Technique used:
• Frequency analysis

