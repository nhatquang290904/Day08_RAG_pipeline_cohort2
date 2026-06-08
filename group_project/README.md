# Bài Tập Nhóm - DrugLaw RAG Chatbot

## Mục Tiêu

Nhóm xây dựng một RAG chatbot trả lời câu hỏi về pháp luật ma túy và tin tức liên quan. Sản phẩm tích hợp pipeline cá nhân từ Task 4 đến Task 10, có giao diện chat, citation, conversation memory, hiển thị source documents và evaluation report.

## Sản Phẩm Đã Làm

- Streamlit chatbot tại `app.py`
- Retrieval pipeline: Task 9
- Generation có citation: Task 10
- PageIndex fallback: Task 8
- Evaluation pipeline: `group_project/evaluation/eval_pipeline.py`
- Golden dataset 15 câu: `group_project/evaluation/golden_dataset.json`
- Báo cáo kết quả: `group_project/evaluation/results.md`

## Kiến Trúc Hệ Thống

```text
User
  |
  v
Streamlit Chat UI (app.py)
  |
  v
Conversation Memory
  |
  v
Task 9 Retrieval Pipeline
  |-- Task 5 Semantic Search
  |-- Task 6 BM25 Lexical Search
  |-- Task 7 Reranking
  |-- Task 8 PageIndex fallback
  |
  v
Task 10 Generation with Citation
  |-- Gemini API when available
  |-- Local citation fallback for offline demo
  |
  v
Answer + Source Documents + Scores
```

## Cách Chạy Chatbot

```bash
pip install -r requirements.txt
streamlit run app.py
```

Trong sidebar:

- `Fast local demo mode`: bật mặc định để demo nhanh, không phụ thuộc network.
- Tắt `Fast local demo mode`: gọi Task 10 đầy đủ, ưu tiên Gemini API nếu môi trường cho phép.
- `Top K sources`: điều chỉnh số source documents đưa vào câu trả lời.

## Cách Chạy Evaluation

```bash
python group_project/evaluation/eval_pipeline.py
```

Script sẽ:

- Load 15 câu hỏi từ golden dataset.
- Chạy 2 config A/B:
  - Config A: hybrid search + rerank.
  - Config B: hybrid search không rerank.
- Tính 4 metric:
  - Faithfulness
  - Answer Relevance
  - Context Recall
  - Context Precision
- Xuất báo cáo vào `group_project/evaluation/results.md`.

## Kết Quả Evaluation Gần Nhất

| Metric | Config A (hybrid + rerank) | Config B (hybrid no-rerank) |
|--------|-----------------------------|------------------------------|
| Faithfulness | 0.932 | 0.933 |
| Answer Relevance | 0.057 | 0.059 |
| Context Recall | 0.233 | 0.317 |
| Context Precision | 0.276 | 0.272 |
| Average | 0.375 | 0.395 |

Kết luận: Config B nhỉnh hơn trong lần chạy offline này. Nguyên nhân chính là dữ liệu pháp luật local còn nhiều placeholder do PDF chưa OCR đầy đủ, khiến recall với một số câu hỏi pháp luật chi tiết chưa cao.

## API Và Fallback

| Thành phần | API chính | Fallback |
|-----------|-----------|----------|
| Task 4 Indexing | Weaviate Cloud nếu upload thành công | Local JSON index |
| Task 7 Reranking | Jina Reranker API | Local token overlap |
| Task 8 Vectorless | PageIndex API | Local lexical fallback |
| Task 10 Generation | Gemini API | Local citation answer |

Thiết kế fallback giúp app vẫn demo được khi network/proxy/API quota không ổn định.

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Thành viên 1 | TBD | Data ingestion, chunking, indexing | Hoàn thành |
| Thành viên 2 | TBD | Semantic/BM25 retrieval, reranking | Hoàn thành |
| Thành viên 3 | TBD | PageIndex fallback, generation citation | Hoàn thành |
| Thành viên 4 | TBD | Streamlit UI, evaluation report | Hoàn thành |

## Checklist Theo README

- [x] Giao diện chat Streamlit
- [x] Trả lời có citation
- [x] Conversation memory cho follow-up questions
- [x] Hiển thị source documents, score và provider
- [x] Golden dataset tối thiểu 15 Q&A
- [x] Evaluation với 4 metric
- [x] A/B comparison tối thiểu 2 configs
- [x] Báo cáo kết quả và phân tích worst performers

## Hạn Chế Và Hướng Cải Tiến

- Cần OCR lại PDF pháp luật để tăng context recall.
- Cần chạy demo trong môi trường network ổn định để bật Gemini/Jina/PageIndex thật.
- Có thể deploy Streamlit lên Hugging Face Spaces hoặc Render để lấy điểm bonus deploy.
