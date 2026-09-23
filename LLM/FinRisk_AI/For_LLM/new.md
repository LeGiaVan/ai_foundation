Đúng. Proposal hiện tại của bạn đang **thiên về “Financial RAG + Multi-Agent + DevOps”**, trong khi sau khi phân rã workflow thẩm định viên, bài toán thực tế lớn hơn nhiều: **Document Intelligence → Financial Fact Extraction → Evidence Retrieval → Financial Calculation → Cross-document Verification → Risk Analysis → Credit Memo**.

 Tôi sẽ viết lại proposal theo hướng đó. Tôi cũng sẽ bỏ một số assumption hơi mạnh như “độ chính xác số liệu 100%”, vì với hệ thống AI tài chính nên đặt mục tiêu là **deterministic calculation + provenance + measurable extraction accuracy**, thay vì cam kết 100%.

 Dưới đây là bản MD có thể dùng thay thế trực tiếp.

 # 🏦 PROJECT PROPOSAL: FinRisk AI — v3.0

 ## Evidence-Aware Financial Intelligence & Credit Risk Assessment System

 ### _Enterprise Financial Intelligence Copilot for Corporate Lending_

---

 # 1\. TỔNG QUAN DỰ ÁN

 ## 1.1. Mục tiêu

 FinRisk AI là hệ thống AI hỗ trợ **thẩm định tín dụng doanh nghiệp**, được thiết kế dựa trên workflow thực tế của chuyên viên tín dụng thay vì chỉ tập trung vào Question Answering trên PDF.

 Hệ thống nhận một **Loan Case** gồm nhiều nguồn dữ liệu:

 - Hồ sơ doanh nghiệp
- Đơn đề nghị cấp tín dụng
- Báo cáo tài chính
- Thuyết minh báo cáo tài chính
- Hồ sơ tài sản bảo đảm
- Lịch sử tín dụng
- Dữ liệu ngân hàng nội bộ
- Thông tin ngành
- Tin tức và thông tin bên ngoài

 Sau đó hệ thống thực hiện:

```
Raw Documents
      ↓
Document Understanding
      ↓
Evidence Extraction
      ↓
Financial Fact Extraction
      ↓
Evidence / Fact Store
      ↓
Hybrid Retrieval
      ↓
Financial Calculation
      ↓
Cross-document Verification
      ↓
Risk Analysis
      ↓
Loan Structure Analysis
      ↓
Credit Memo
```

 Mục tiêu cuối cùng không phải là tạo một chatbot đọc BCTC.

 Mục tiêu là xây dựng:

 > **Một Financial Intelligence Copilot có khả năng biến hồ sơ tín dụng phi cấu trúc thành các financial facts, evidence, metrics và risk signals có thể kiểm chứng và truy xuất nguồn gốc.**

---

 # 2\. BÀI TOÁN THỰC TẾ

 ## 2.1. Workflow của chuyên viên thẩm định

 Một case vay doanh nghiệp thường có workflow:

```
1. Tiếp nhận hồ sơ
        ↓
2. Xác minh doanh nghiệp
        ↓
3. Hiểu nhu cầu vay
        ↓
4. Phân tích hoạt động kinh doanh
        ↓
5. Phân tích BCTC
        ↓
6. Phân tích khả năng trả nợ
        ↓
7. Phân tích lịch sử tín dụng
        ↓
8. Phân tích TSBĐ
        ↓
9. Phân tích ngành / thị trường
        ↓
10. Tra cứu thông tin bên ngoài
        ↓
11. Kiểm tra chéo toàn bộ hồ sơ
        ↓
12. Xác định các rủi ro chính
        ↓
13. Xác định biện pháp giảm thiểu rủi ro
        ↓
14. Phân tích cấu trúc khoản vay
        ↓
15. Lập Credit Memo
        ↓
16. Human Credit Approval
```

 FinRisk AI không thay thế toàn bộ quyết định tín dụng.

 Hệ thống cung cấp:

```
Evidence
+
Financial Facts
+
Calculations
+
Risk Signals
+
Explainable Analysis
+
Recommended Investigation Points
```

 để chuyên viên tín dụng đưa ra quyết định cuối cùng.

---

 # 3\. RESEARCH & PRODUCT PROBLEM

 ## 3.1. Vấn đề của RAG truyền thống

 Financial documents có cấu trúc đặc biệt:

```
PDF
├── Narrative
├── Financial Tables
├── Notes
├── Footnotes
├── Cross references
├── Multiple reporting periods
└── Numerical relationships
```

 Nếu áp dụng:

```
PDF
→ 500-token chunks
→ Embedding
→ Vector Search
→ LLM
```

 sẽ xảy ra các vấn đề:

 - Mất header của bảng.
- Mất mapping giữa chỉ tiêu và kỳ báo cáo.
- Tách công thức khỏi các biến đầu vào.
- Không phân biệt số liệu năm hiện tại và năm trước.
- Không truy nguyên được một metric về cell/row/page gốc.
- Không phù hợp với numerical reasoning.
- Không phát hiện được inconsistency giữa nhiều hồ sơ.

 Do đó FinRisk AI sử dụng:

 > **Evidence-Aware Hierarchical Financial RAG**

 thay vì chỉ sử dụng token-based RAG.

---

 # 4\. CORE DESIGN PRINCIPLE

 ## 4.1. Chunk không phải Data Model

 Trong FinRisk AI:

```
Chunk
```

 chỉ là một representation phục vụ retrieval.

 Data model chính là:

```
Document
Section
Evidence
Table
FinancialFact
Entity
Relationship
FinancialMetric
RiskSignal
LoanFacility
```

 Quan hệ:

```
Document
   ↓
Evidence
   ↓
FinancialFact
   ↓
FinancialMetric
   ↓
RiskSignal
   ↓
Credit Memo
```

 Ví dụ:

```
BCTC.pdf
   ↓
Page 25
   ↓
Balance Sheet
   ↓
Current Assets
   ↓
Current Assets = 500B
   ↓
Current Ratio = 0.82
   ↓
Liquidity Risk Signal
   ↓
Credit Memo
```

---

 # 5\. INPUT UNIVERSE

 ## 5.1. Borrower Information

 ### Input

 - Tên doanh nghiệp
- Mã số thuế
- Giấy đăng ký kinh doanh
- Loại hình pháp lý
- Ngày thành lập
- Địa chỉ
- Ngành nghề
- Công ty mẹ
- Công ty con
- Cổ đông
- Ultimate Beneficial Owner
- Ban lãnh đạo
- Người đại diện pháp luật

 ### Processing

```
Document Extraction
+
Entity Resolution
+
Relationship Mapping
```

 ### Output

```
Borrower Profile
Ownership Structure
Management Profile
Relationship Graph
```

---

 # 6\. LOAN APPLICATION

 ## 6.1. Input

```
Requested Amount
Currency
Facility Type
Tenor
Purpose
Interest Rate
Repayment Schedule
Repayment Source
Disbursement Plan
Existing Facilities
Proposed Collateral
```

 ## 6.2. Output

```
Loan Request Object
```

 Ví dụ:

```
{
  "requested_amount": 50000000000,
  "purpose": "working_capital",
  "tenor_months": 12,
  "repayment_source": "operating_cash_flow"
}
```

---

 # 7\. FINANCIAL STATEMENT PROCESSING

 ## 7.1. Các nguồn

```
Balance Sheet
Income Statement
Cash Flow Statement
Notes
Accounting Policies
Audit Report
```

 ## 7.2. Balance Sheet Facts

```
Cash
Accounts Receivable
Inventory
Other Current Assets
Fixed Assets
Long-term Investments
Other Assets

Accounts Payable
Short-term Debt
Long-term Debt
Other Liabilities

Equity
Retained Earnings
```

 ## 7.3. Income Statement Facts

```
Revenue
COGS
Gross Profit
Operating Expenses
EBIT
EBITDA
Interest Expense
EBT
Tax
Net Income
```

 ## 7.4. Cash Flow Facts

```
CFO
CFI
CFF
Capex
Debt Proceeds
Debt Repayment
Dividends
```

---

 # 8\. FINANCIAL EVIDENCE MODEL

 Financial table không được lưu đơn thuần như text chunk.

 ## 8.1. Table Object

```
Financial Table
├── Table Metadata
├── Periods
├── Rows
├── Columns
└── Source Location
```

 ## 8.2. Financial Fact

 Mỗi số liệu quan trọng được chuẩn hóa thành:

```
{
  "fact_id": "FACT_001",
  "concept": "total_assets",
  "value": 10000000000,
  "currency": "VND",
  "period": "2025",
  "unit": "VND",
  "document_id": "BCTC_2025",
  "page": 5,
  "table_id": "TABLE_003",
  "row": "Tổng tài sản",
  "confidence": 0.98
}
```

 ## 8.3. Provenance

 Mọi Financial Fact phải truy nguyên được:

```
Fact
 ↓
Table
 ↓
Page
 ↓
PDF
```

 Đây là yêu cầu bắt buộc đối với các số liệu được sử dụng để tính toán.

---

 # 9\. FINANCIAL ONTOLOGY

 Hệ thống xây dựng canonical financial concepts.

 Ví dụ:

```
"Doanh thu thuần"
"Doanh thu bán hàng"
"Revenue"
"Net revenue"
```

 →

```
REVENUE
```

 Tương tự:

```
"EBIT"
"Lợi nhuận trước lãi vay và thuế"
"Operating profit"
```

 →

```
EBIT
```

 Ontology giúp:

 - Chuẩn hóa tên chỉ tiêu.
- Retrieval xuyên nhiều BCTC.
- Tính financial ratios.
- Phát hiện inconsistency.
- Tái sử dụng formula.
- Hỗ trợ multi-company comparison.

---

 # 10\. CHUNKING STRATEGY

 FinRisk AI sử dụng nhiều loại evidence representation.

 ## 10.1. Narrative Evidence

```
Section
    ↓
Paragraph
    ↓
Semantic Evidence Unit
```

 Không ép mọi chunk có cùng token size.

---

 ## 10.2. Financial Table Evidence

```
Table
├── Table-level Evidence
├── Row Evidence
├── Cell Evidence
└── Financial Facts
```

 Ví dụ:

```
Total Assets
2025 = 10,000B
2024 = 9,000B
```

 được lưu dưới dạng:

```
Table Evidence
+
Row Evidence
+
Financial Facts
```

---

 ## 10.3. Application Evidence

 Đơn đề nghị được chuyển thành:

```
Loan Application Fields
+
Original Evidence
```

---

 ## 10.4. Collateral Evidence

 Hồ sơ TSBĐ được chuyển thành:

```
Collateral Object
+
Legal Evidence
+
Valuation Evidence
+
Ownership Evidence
```

---

 ## 10.5. News Evidence

 Tin tức được chuyển thành:

```
News Event
+
Entity
+
Date
+
Source
+
Claim
+
Evidence
```

---

 # 11\. MULTI-VIEW RETRIEVAL

 Một Evidence Object có thể có nhiều representation:

```
Raw Text
Semantic Chunk
Financial Fact
Synthetic Query
Keywords
Metadata
```

 Từ đó xây dựng:

```
Dense Index
+
Sparse Index
+
Structured Fact Store
```

---

 # 12\. HYBRID RETRIEVAL

 Pipeline:

```
User Query
    ↓
Query Classification
    ↓
Query Expansion
    ↓
Dense Retrieval
+
Sparse Retrieval
+
Structured Retrieval
    ↓
Fusion
    ↓
Cross Encoder Reranker
    ↓
Evidence Expansion
    ↓
Top Evidence
```

 Ví dụ:

 > "Current ratio năm 2025 là bao nhiêu?"

 Structured retrieval:

```
current_assets
current_liabilities
period=2025
```

 sau đó:

```
Current Ratio =
Current Assets / Current Liabilities
```

---

 # 13\. FINANCIAL CALCULATION ENGINE

 LLM không trực tiếp tính toán các financial metrics.

 Architecture:

```
Financial Facts
      ↓
Formula Registry
      ↓
Deterministic Python Calculator
      ↓
Financial Metric
      ↓
Provenance
```

 ## 13.1. Liquidity

```
Current Ratio
Quick Ratio
Cash Ratio
Working Capital
```

 ## 13.2. Leverage

```
Debt / Equity
Debt / Assets
Liabilities / Assets
Net Debt / EBITDA
```

 ## 13.3. Profitability

```
Gross Margin
EBIT Margin
Net Margin
ROA
ROE
EBITDA Margin
```

 ## 13.4. Efficiency

```
DSO
DIO
DPO
Cash Conversion Cycle
Asset Turnover
```

 ## 13.5. Debt Service

```
DSCR
Interest Coverage
Debt Capacity
Debt Service
```

 ## 13.6. Financial Risk Models

```
Altman Z-Score
```

 và các mô hình/ratio khác có thể được bổ sung thông qua Formula Registry.

---

 # 14\. FINANCIAL METRIC PROVENANCE

 Ví dụ:

```
Altman Z-Score
       │
       ├── X1
       │    ├── Working Capital
       │    │     ├── Current Assets
       │    │     └── Current Liabilities
       │    └── Total Assets
       │
       ├── X2
       ├── X3
       ├── X4
       └── X5
```

 Mỗi metric phải trả lời được:

 > Metric này được tính từ những số liệu nào?

 và:

 > Những số liệu đó lấy từ đâu trong hồ sơ?

---

 # 15\. REPAYMENT CAPACITY ENGINE

 Một module riêng đánh giá khả năng trả nợ.

 ## Input

```
Requested Loan
Existing Debt
Interest Expense
Principal Repayment
Operating Cash Flow
Free Cash Flow
Revenue
EBITDA
Working Capital Requirement
```

 ## Output

```
Debt Service
DSCR
Interest Coverage
Cash Flow Surplus
Debt Capacity
```

---

 # 16\. WORKING CAPITAL ANALYSIS

 Đối với khoản vay vốn lưu động:

```
Revenue
COGS
Receivables
Inventory
Payables
```

 →

```
DSO
DIO
DPO
Cash Conversion Cycle
Working Capital Requirement
```

 Sau đó:

```
Required Working Capital
        vs
Requested Loan
```

 Hệ thống phải xác định:

 > Khoản vay được đề nghị có phù hợp với nhu cầu vốn lưu động được quan sát từ hoạt động kinh doanh hay không?

---

 # 17\. CREDIT HISTORY

 ## Input

```
Existing Loans
Outstanding Balance
Credit Limits
Overdue
Restructuring
Repayment History
Other Banks
Guarantees
Off-balance-sheet Obligations
```

 ## Output

```
Credit History Evidence
Credit Risk Signals
Existing Debt Profile
Repayment Behavior
```

---

 # 18\. COLLATERAL INTELLIGENCE

 ## 18.1. Collateral Object

```
Collateral
├── Asset Type
├── Owner
├── Legal Status
├── Location
├── Size
├── Condition
├── Market Value
├── Appraised Value
├── Forced-sale Value
├── Valuation Date
├── Existing Encumbrance
├── Insurance
└── Liquidity
```

 ## 18.2. Analysis

```
Legal Validity
+
Ownership
+
Valuation
+
Liquidity
+
Coverage
```

 ## 18.3. Metrics

```
LTV
Collateral Coverage
Haircut-adjusted Coverage
```

---

 # 19\. INDUSTRY & BUSINESS ANALYSIS

 ## Input

```
Industry
Market
Competition
Customer Concentration
Supplier Concentration
Regulation
Commodity Exposure
FX Exposure
Interest Rate Exposure
Cyclicality
```

 ## Processing

```
Internal Documents
+
External Data
+
Web Search
+
LLM Analysis
```

 ## Output

```
Industry Risk Signals
Business Model Risks
Market Risks
Regulatory Risks
```

---

 # 20\. EXTERNAL INFORMATION AGENT

 External information có thể sử dụng Web Search / Search API.

 Agent tìm:

```
Company News
Legal Disputes
Tax Issues
Regulatory Violations
Management News
Major Customer Events
Major Supplier Events
Industry Events
M&A
Reputation Signals
```

 Pipeline:

```
Search
 ↓
Source Validation
 ↓
Entity Resolution
 ↓
Publication Date
 ↓
Fact Extraction
 ↓
Evidence
 ↓
Risk Signal
```

 Mỗi external signal phải lưu:

```
Source
URL
Publication Date
Entity
Claim
Evidence
Confidence
```

---

 # 21\. CROSS-DOCUMENT CONSISTENCY AGENT

 Đây là một module quan trọng.

 Hệ thống so sánh:

```
Loan Application
        ↕
Financial Statements
        ↕
Credit Information
        ↕
Collateral Documents
        ↕
External Information
```

 Ví dụ:

```
Application:
Revenue = 1,000B

Financial Statement:
Revenue = 700B
```

 →

```
Inconsistency Signal
```

 Ví dụ:

```
Application:
Purpose = Inventory

Financial Statement:
Inventory ↓ 30%
```

 →

```
Investigation Signal
```

 Ví dụ:

```
Financial Statement:
Debt = 100B

Credit Information:
Debt = 250B
```

 →

```
Material Debt Discrepancy
```

---

 # 22\. RELATED-PARTY ANALYSIS

 Xây dựng relationship graph:

```
Company
├── Shareholder
├── Director
├── Parent
├── Subsidiary
├── Related Company
├── Customer
└── Supplier
```

 Phân tích:

```
Related-party Receivables
Related-party Loans
Related-party Revenue
Related-party Transactions
```

---

 # 23\. 5C RISK FRAMEWORK

 FinRisk AI sử dụng 5C làm framework tổng hợp.

 ## Character

```
Management
Credit History
Repayment Behavior
Legal Issues
External Reputation
Related-party Behavior
```

 ## Capacity

```
Revenue
Profitability
Cash Flow
Debt Service
DSCR
Debt Capacity
```

 ## Capital

```
Equity
Debt / Equity
Owner Contribution
Retained Earnings
Capital Structure
```

 ## Collateral

```
Ownership
Legal Status
Value
Liquidity
LTV
Coverage
```

 ## Conditions

```
Industry
Market
Regulation
Competition
Macroeconomic Conditions
External Events
```

---

 # 24\. RISK REGISTER

 Thay vì output duy nhất:

```
Risk = High
```

 hệ thống tạo:

```
{
  "risk_id": "R001",
  "category": "Liquidity",
  "description": "Receivables concentration",
  "evidence": [
    "BCTC page 24",
    "Note 12"
  ],
  "impact": "...",
  "trigger": "...",
  "mitigant": "...",
  "confidence": 0.91
}
```

 Risk Register gồm:

```
Risk
├── Category
├── Description
├── Evidence
├── Impact
├── Trigger
├── Confidence
├── Mitigant
└── Monitoring Indicator
```

---

 # 25\. RISK ASSESSMENT AGENT

 Risk Agent nhận:

```
Financial Metrics
+
Financial Facts
+
Credit History
+
Collateral
+
Industry Signals
+
External News
+
Consistency Signals
```

 và tạo:

```
Risk Register
+
Key Strengths
+
Key Weaknesses
+
Investigation Points
+
Risk Mitigants
```

 Agent **không được tự tạo số liệu**.

 Mọi numerical claim phải truy xuất được về:

```
FinancialFact
```

 hoặc:

```
ExternalEvidence
```

---

 # 26\. LOAN STRUCTURING

 Input:

```
Borrower Risk
Repayment Capacity
Collateral Coverage
Existing Exposure
Loan Purpose
Cash Flow
```

 Output:

```
Loan Amount
Tenor
Repayment Structure
Collateral
LTV
Covenants
Disbursement Conditions
Monitoring Conditions
```

 Hệ thống có thể đưa ra **recommendation để chuyên viên xem xét**, nhưng quyết định tín dụng cuối cùng thuộc quy trình phê duyệt của ngân hàng.

---

 # 27\. CREDIT MEMO GENERATION

 Credit Memo:

```
1. Executive Summary
2. Borrower Profile
3. Loan Request
4. Business Model
5. Management
6. Industry
7. Financial Analysis
8. Cash Flow Analysis
9. Debt Capacity
10. Credit History
11. Collateral
12. External Information
13. Consistency Findings
14. Key Risks
15. Risk Mitigants
16. Proposed Loan Structure
17. Monitoring Conditions
```

 Mỗi assertion quan trọng phải có:

```
Claim
 ↓
Evidence
 ↓
Source
 ↓
Page / Table / Row / Cell
```

---

 # 28\. KIẾN TRÚC HỆ THỐNG

```
flowchart TD

    subgraph INPUT["📂 Loan Case Inputs"]
        Application["Loan Application"]
        Financial["Financial Statements"]
        Collateral["Collateral Documents"]
        Company["Company Documents"]
        Credit["Credit History"]
        External["External Information"]
    end

    subgraph INGESTION["📥 Document Intelligence"]
        Parser["PDF / Document Parser"]
        TableParser["Financial Table Parser"]
        OCR["OCR"]
        EntityExtractor["Entity Extraction"]
    end

    subgraph EVIDENCE["🧩 Evidence Layer"]
        Evidence["Evidence Store"]
        Facts["Financial Fact Store"]
        Ontology["Financial Ontology"]
        Graph["Entity / Relationship Graph"]
    end

    subgraph RETRIEVAL["🔎 Retrieval Layer"]
        Dense["Dense Retrieval"]
        Sparse["Sparse Retrieval"]
        Structured["Structured Retrieval"]
        Reranker["Cross Encoder Reranker"]
    end

    subgraph ANALYTICS["🧮 Financial Analytics"]
        Formula["Formula Engine"]
        Ratios["Financial Ratios"]
        ZScore["Altman Z-Score"]
        DSCR["DSCR / Debt Capacity"]
        WorkingCapital["Working Capital"]
    end

    subgraph AGENTS["🤖 Intelligence Agents"]
        FinancialAgent["Financial Analysis Agent"]
        CollateralAgent["Collateral Agent"]
        CreditAgent["Credit History Agent"]
        IndustryAgent["Industry Agent"]
        NewsAgent["External Information Agent"]
        Consistency["Consistency Agent"]
        RiskAgent["Risk Assessment Agent"]
        MemoAgent["Credit Memo Agent"]
    end

    subgraph OUTPUT["📋 Outputs"]
        RiskRegister["Risk Register"]
        Metrics["Financial Metrics"]
        EvidenceReport["Evidence Report"]
        CreditMemo["Credit Memo"]
        HITL["Human Credit Review"]
    end

    Application --> Parser
    Financial --> Parser
    Collateral --> Parser
    Company --> Parser
    Credit --> Parser
    External --> NewsAgent

    Parser --> OCR
    Parser --> TableParser
    Parser --> EntityExtractor

    OCR --> Evidence
    TableParser --> Facts
    EntityExtractor --> Ontology
    EntityExtractor --> Graph

    Evidence --> Dense
    Evidence --> Sparse
    Facts --> Structured

    Dense --> Reranker
    Sparse --> Reranker
    Structured --> Reranker

    Facts --> Formula
    Ontology --> Formula

    Formula --> Ratios
    Formula --> ZScore
    Formula --> DSCR
    Formula --> WorkingCapital

    Reranker --> FinancialAgent
    Facts --> FinancialAgent
    Ratios --> FinancialAgent
    ZScore --> FinancialAgent
    DSCR --> FinancialAgent

    Collateral --> CollateralAgent
    Credit --> CreditAgent
    External --> NewsAgent

    FinancialAgent --> Consistency
    CollateralAgent --> Consistency
    CreditAgent --> Consistency
    NewsAgent --> Consistency
    Application --> Consistency

    Consistency --> RiskAgent
    FinancialAgent --> RiskAgent
    CollateralAgent --> RiskAgent
    NewsAgent --> RiskAgent

    RiskAgent --> RiskRegister
    RiskAgent --> MemoAgent
    Evidence --> MemoAgent
    Ratios --> MemoAgent
    ZScore --> MemoAgent
    DSCR --> MemoAgent

    MemoAgent --> CreditMemo
    CreditMemo --> HITL
```

---

 # 29\. MULTI-AGENT ARCHITECTURE

 Sử dụng LangGraph StateGraph.

```
                    Supervisor
                         │
        ┌────────────────┼─────────────────┐
        │                │                 │
        ▼                ▼                 ▼
 Financial Agent   Collateral Agent   Credit Agent
        │                │                 │
        └────────────────┼─────────────────┘
                         │
                  Consistency Agent
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   Industry Agent   News Agent      Risk Agent
        │                │                │
        └────────────────┼────────────────┘
                         │
                   Credit Memo
                         │
                         ▼
                    Human Review
```

---

 # 30\. STATE MODEL

 LangGraph State:

```
class CreditCaseState(TypedDict):

    case_id: str

    borrower: dict
    loan_application: dict

    documents: list
    evidence: list
    financial_facts: list

    financial_metrics: dict
    collateral: dict

    credit_history: dict
    industry_analysis: dict
    external_signals: list

    inconsistencies: list
    risk_signals: list

    loan_structure: dict

    credit_memo: dict

    citations: list

    human_review_required: bool
```

---

 # 31\. STORAGE ARCHITECTURE

 ## PostgreSQL

 Lưu:

```
Users
Loan Cases
Borrowers
Loan Applications
Financial Facts
Financial Metrics
Risk Signals
Audit Logs
Agent Runs
```

 ## Qdrant

 Lưu:

```
Narrative Evidence
Table Evidence
Synthetic Queries
Document Metadata
```

 Payload:

```
evidence_id
document_id
case_id
borrower_id
page
section
evidence_type
table_id
concept
period
```

 ## Object Storage

 Lưu:

```
Original PDF
OCR Output
Parsed Documents
Generated Reports
Snapshots
```

 ## Optional Graph Layer

 Có thể sử dụng PostgreSQL relationships trước.

 Graph DB chỉ được thêm khi relationship traversal trở thành bottleneck thực tế.

---

 # 32\. TECH STACK

 | Tầng | Công nghệ | Mục đích |
| --- | --- | --- |
| API | FastAPI | REST/SSE |
| Agent Orchestration | LangGraph | StateGraph / HITL |
| LLM | Model API phù hợp | Reasoning / Extraction |
| Embedding | BGE-M3 hoặc model phù hợp | Dense Retrieval |
| Sparse | BM25 | Exact Retrieval |
| Vector DB | Qdrant | Evidence Retrieval |
| Reranker | BGE Reranker | Retrieval Precision |
| Database | PostgreSQL | Structured Data |
| Object Storage | S3/R2/MinIO | Original Documents |
| Cache | Redis | Cache / Queue |
| Worker | Celery | Background Processing |
| PDF | pdfplumber / PyMuPDF | Parsing |
| OCR | PaddleOCR / equivalent | Scanned PDFs |
| Calculation | Python | Deterministic Finance Engine |
| Web Intelligence | Search API | External Information |
| Observability | Langfuse | Tracing / Cost |
| Evaluation | Ragas / DeepEval + custom metrics | Evaluation |
| Container | Docker Compose | Deployment |
| Proxy | Nginx | Reverse Proxy |
| CI/CD | GitHub Actions | Automated Deployment |
| Infrastructure | VPS | Deployment |

---

 # 33\. EVALUATION FRAMEWORK

 Evaluation không chỉ đo RAG answer quality.

 ## 33.1. Document Extraction

```
Table Extraction Accuracy
OCR Accuracy
Field Extraction F1
Entity Extraction F1
```

 ## 33.2. Financial Fact Extraction

```
Concept Accuracy
Value Accuracy
Period Accuracy
Unit Accuracy
Provenance Accuracy
```

 ## 33.3. Retrieval

```
Recall@K
Precision@K
MRR
nDCG
Evidence Recall
Evidence Precision
```

 ## 33.4. Numerical Reasoning

```
Numeric Accuracy
Formula Accuracy
Intermediate Fact Accuracy
```

 ## 33.5. Risk Analysis

```
Risk Signal Precision
Risk Evidence Coverage
Unsupported Claim Rate
```

 ## 33.6. Generation

```
Faithfulness
Citation Accuracy
Citation Completeness
Hallucination Rate
```

---

 # 34\. RESEARCH EXPERIMENT

 Một mục tiêu nghiên cứu chính:

 > **Can evidence-aware hierarchical retrieval improve financial fact retrieval and credit-risk analysis compared with conventional token-based RAG?**

 ## Baselines

```
S1 — Fixed-size Chunking
S2 — Recursive Chunking
S3 — Table-preserving Chunking
S4 — Parent-Child RAG
S5 — Semantic Chunking
S6 — Hybrid Retrieval
S7 — Evidence-Aware Hierarchical RAG
S8 — Evidence-Aware RAG + Financial Fact Engine
```

---

 # 35\. ABLATION STUDY

 So sánh:

```
Dense only
vs
BM25 only
vs
Hybrid
```

 và:

```
Raw Chunk
vs
Semantic Evidence
vs
Financial Fact
vs
Evidence + Fact
```

 và:

```
RAG-only Calculation
vs
LLM Calculation
vs
Deterministic Calculation
```

 Mục tiêu là chứng minh:

 > Financial reasoning không nên chỉ được thực hiện bằng text generation.

---

 # 36\. GOLDEN DATASET

 Xây dựng bộ testset riêng cho Credit Risk.

 Mỗi sample:

```
{
  "case_id": "CASE_001",
  "question": "Current ratio năm 2025 là bao nhiêu?",
  "required_facts": [
    "current_assets",
    "current_liabilities"
  ],
  "gold_evidence": [
    "BCTC_2025:p5:table3"
  ],
  "formula": "current_assets / current_liabilities",
  "gold_answer": 0.82
}
```

 Các nhóm benchmark:

```
Financial Fact Retrieval
Financial Ratio
Cash Flow
Debt Capacity
Collateral
Cross-document Consistency
External Risk
5C Analysis
```

---

 # 37\. LỘ TRÌNH 12 TUẦN

 # 🔵 PHASE 1 — Tuần 1–2

 ## Credit Workflow + Financial Data Model

 ### Mục tiêu

 Hiểu workflow thẩm định và thiết kế canonical data model.

 ### Deliverables

 - Loan Case schema
- Borrower schema
- Loan Application schema
- Evidence schema
- Financial Fact schema
- Financial Metric schema
- Risk Signal schema
- Provenance model

 ### Kết quả

```
Raw Document
→ Structured Case Model
```

---

 # 🟢 PHASE 2 — Tuần 3–4

 ## Document Intelligence + Evidence-Aware RAG

 ### Mục tiêu

 Xây ingestion pipeline.

```
PDF
 ↓
Parser/OCR
 ↓
Section Detection
 ↓
Table Detection
 ↓
Evidence Extraction
 ↓
Financial Fact Extraction
 ↓
Index
```

 ### Experiments

 So sánh:

```
Recursive
Table-preserving
Semantic
Parent-child
Evidence-aware
```

 ### Deliverables

 - PDF parser
- Table parser
- Evidence builder
- Financial Fact extractor
- Qdrant indexing
- BM25
- Hybrid retrieval
- Reranker

---

 # 🟡 PHASE 3 — Tuần 5–6

 ## Financial Intelligence Engine

 ### Mục tiêu

 Xây deterministic financial calculation layer.

 ### Implement

```
Liquidity
Profitability
Leverage
Efficiency
Working Capital
DSCR
Debt Capacity
Altman Z-Score
```

 ### Yêu cầu

 Mọi metric phải có:

```
Formula
Inputs
Values
Source
Provenance
```

---

 # 🟠 PHASE 4 — Tuần 7–8

 ## Multi-Agent Credit Analysis

 ### Agents

```
Financial Agent
Collateral Agent
Credit History Agent
Industry Agent
External Information Agent
Consistency Agent
Risk Agent
Credit Memo Agent
```

 ### Orchestration

```
Supervisor
 ↓
Parallel Analysis
 ↓
Consistency
 ↓
Risk
 ↓
Memo
```

---

 # 🔴 PHASE 5 — Tuần 9–10

 ## Production Infrastructure

 ### Mục tiêu

 Deploy hệ thống.

```
FastAPI
Docker
Redis
Celery
PostgreSQL
Qdrant
Nginx
SSL
GitHub Actions
```

 ### Security

```
JWT
Rate Limiting
Secrets
Input Validation
Audit Logging
```

---

 # 🟣 PHASE 6 — Tuần 11–12

 ## Evaluation + Observability + Research

 ### Evaluation

```
Ragas
DeepEval
Custom Financial Metrics
Golden Dataset
Ablation Study
```

 ### Observability

```
Langfuse
Token Cost
Latency
Retrieval Trace
Agent Trace
Calculation Trace
```

 ### Research Output

```
Experiment Results
Ablation Table
Error Analysis
Architecture Diagram
Research Report
```

---

 # 38\. PRODUCTION ARCHITECTURE

 Mermaid flowchart: 👤 Credit Analyst, 🌐 Internet, Nginx / SSL, FastAPI, Auth + RBAC, Redis / Celery, LangGraph Supervisor, Financial Agent, Collateral Agent, Credit Agent, Industry Agent, External Information Agent, Consistency Agent, Risk Agent, Credit Memo Agent, ("PostgreSQL"), ("Qdrant"), ("Object Storage"), ("Redis"), Langfuse

---

 # 39\. OBSERVABILITY

 Mỗi request phải trace được:

```
Case
 ↓
Agent
 ↓
Retrieval
 ↓
Evidence
 ↓
Financial Fact
 ↓
Formula
 ↓
Metric
 ↓
Risk Signal
 ↓
Credit Memo
```

 Langfuse lưu:

```
Trace
Span
Prompt
Token Usage
Latency
Model
Cost
Retrieved Evidence
Agent Decision
```

---

 # 40\. HUMAN-IN-THE-LOOP

 FinRisk AI không tự động phê duyệt khoản vay.

 Human review được yêu cầu khi:

```
Material Data Inconsistency
OR
Missing Critical Evidence
OR
Low Extraction Confidence
OR
High Risk Signal
OR
Calculation Input Ambiguous
OR
External Information Conflict
```

 UI phải cho phép:

```
Accept
Reject
Request Investigation
Override
Add Comment
```

 và lưu:

```
Reviewer
Timestamp
Decision
Reason
Evidence
```

---

 # 41\. AUDITABILITY

 Một kết luận:

```
"Liquidity risk detected"
```

 phải truy ngược được:

```
Risk Signal
 ↓
Current Ratio
 ↓
Current Assets
 ↓
Current Liabilities
 ↓
BCTC
 ↓
Page
 ↓
Table
 ↓
Row
 ↓
Original PDF
```

 Đây là một trong những yêu cầu kiến trúc cốt lõi của hệ thống.

---

 # 42\. SECURITY

 ## Data Security

```
Encryption in transit
Encryption at rest
Document access control
Tenant isolation
```

 ## Application Security

```
JWT
RBAC
Rate Limiting
Input Validation
File Type Validation
File Size Limit
Prompt Injection Protection
```

 ## AI Security

```
Prompt Injection Detection
Tool Permission Control
LLM Output Validation
Financial Calculation Isolation
Citation Validation
```

---

 # 43\. COST & PERFORMANCE

 Theo dõi:

```
Cost / Loan Case
Cost / Document
Cost / Agent
Cost / LLM Call
Tokens / Case
Latency / Case
```

 Cache:

```
Document Parsing
Embeddings
Retrieval
Repeated Queries
```

 Không cache các kết quả phụ thuộc vào dữ liệu đã thay đổi mà không có versioning.

---

 # 44. DEFINITION OF DONE

 ## Functional

 - [ ] Upload Loan Case
- [ ] Parse multiple PDFs
- [ ] Extract financial tables
- [ ] Extract Financial Facts
- [ ] Store provenance
- [ ] Hybrid Retrieval
- [ ] Reranking
- [ ] Financial Calculation Engine
- [ ] Altman Z-Score
- [ ] DSCR
- [ ] Working Capital Analysis
- [ ] Collateral Analysis
- [ ] External Information Agent
- [ ] Cross-document Consistency
- [ ] Risk Register
- [ ] Credit Memo

 ## Research

 - [ ] Fixed Chunk baseline
- [ ] Recursive baseline
- [ ] Table-preserving baseline
- [ ] Parent-child baseline
- [ ] Evidence-aware RAG
- [ ] Ablation study
- [ ] Golden dataset
- [ ] Retrieval evaluation
- [ ] Financial fact evaluation
- [ ] Numerical accuracy evaluation
- [ ] Risk evidence evaluation

 ## Production

 - [ ] Docker Compose
- [ ] FastAPI
- [ ] PostgreSQL
- [ ] Qdrant
- [ ] Redis
- [ ] Celery
- [ ] Nginx
- [ ] HTTPS
- [ ] GitHub Actions
- [ ] Langfuse
- [ ] Structured Logging
- [ ] Backup
- [ ] Health Check
- [ ] RBAC
- [ ] Audit Trail

---

 # 45\. SUCCESS METRICS

 Không sử dụng một metric duy nhất như:

```
Faithfulness > 0.85
```

 Hệ thống được đánh giá theo nhiều tầng.

 | Layer | Metric |
| --- | --- |
| OCR | Character / Word Accuracy |
| Extraction | Precision / Recall / F1 |
| Financial Fact | Concept Accuracy |
| Financial Fact | Value Accuracy |
| Financial Fact | Period Accuracy |
| Provenance | Citation Accuracy |
| Retrieval | Recall@K |
| Retrieval | MRR / nDCG |
| Reranking | Precision@K |
| Calculation | Numeric Accuracy |
| Calculation | Formula Accuracy |
| Risk | Evidence Coverage |
| Risk | Unsupported Claim Rate |
| Generation | Faithfulness |
| System | p95 Latency |
| System | Cost / Case |

---

 # 46\. FINAL PRODUCT

 Một Loan Case sau khi chạy qua FinRisk AI sẽ tạo ra:

```
                 FINRISK AI
                     │
        ┌────────────┼─────────────┐
        │            │             │
        ▼            ▼             ▼
 Financial       Collateral     External
 Analysis        Analysis       Intelligence
        │            │             │
        └────────────┼─────────────┘
                     │
              Consistency Check
                     │
                     ▼
               Risk Register
                     │
                     ▼
             Loan Structure
                     │
                     ▼
               Credit Memo
                     │
                     ▼
             👤 Human Review
```

 Credit analyst có thể xem:

```
Financial Summary
       +
Financial Metrics
       +
Risk Signals
       +
Inconsistencies
       +
Collateral
       +
External Information
       +
Evidence Citations
       +
Calculation Provenance
```

 thay vì phải đọc toàn bộ hồ sơ từ đầu để tìm từng thông tin.

---

 # 47\. PROJECT POSITIONING

 FinRisk AI không được định vị đơn thuần là:

 > "A RAG chatbot for financial reports."

 Mà là:

 > **An Evidence-Aware Financial Intelligence System for Corporate Credit Risk Assessment.**

 Các technical contributions chính:

 ### Contribution 1 — Evidence-Aware Financial Representation

 Thay vì token chunks:

```
Document
→ Evidence
→ Financial Fact
```

 ### Contribution 2 — Hierarchical Financial Retrieval

 Kết hợp:

```
Dense Retrieval
+
Sparse Retrieval
+
Structured Fact Retrieval
+
Reranking
```

 ### Contribution 3 — Deterministic Financial Reasoning

 LLM tìm evidence.

 Python Formula Engine tính toán.

```
Evidence
→ Fact
→ Formula
→ Metric
```

 ### Contribution 4 — Cross-Document Risk Detection

```
Application
↔
Financial Statement
↔
Credit History
↔
Collateral
↔
External Information
```

 ### Contribution 5 — Explainable Credit Analysis

 Mọi risk signal đều truy nguyên được:

```
Risk
→ Metric
→ Fact
→ Evidence
→ Source
→ Page / Table / Row
```

---

 # 48\. END-TO-END SYSTEM

```
                         LOAN CASE
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
   Application             BCTC                 Collateral
       │                     │                     │
       │                PDF/OCR Parser             │
       │                     │                     │
       │              Table / Narrative            │
       │                     │                     │
       └──────────────┬──────┴──────────┬──────────┘
                      │                 │
                      ▼                 ▼
                  Evidence        Financial Facts
                      │                 │
                      └────────┬────────┘
                               │
                    Financial Ontology
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
             Vector          BM25         Fact Store
                │              │              │
                └──────────────┼──────────────┘
                               │
                         Hybrid Retrieval
                               │
                           Reranker
                               │
                               ▼
                      Financial Analysis
                               │
               ┌───────────────┼────────────────┐
               │               │                │
               ▼               ▼                ▼
            Ratios          Z-Score          DSCR
               │               │                │
               └───────────────┼────────────────┘
                               │
                  Cross-document Verification
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
      Credit History       External News       Industry
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                         Risk Register
                               │
                         Loan Structure
                               │
                         Credit Memo
                               │
                         👤 HUMAN REVIEW
```

---

 # 49\. KẾT LUẬN

 FinRisk AI được xây dựng theo nguyên tắc:

 > **Retrieve evidence, extract facts, calculate deterministically, reason transparently, and keep a complete audit trail.**

 Hệ thống không xem PDF là một tập các chunks.

 Hệ thống xem hồ sơ tín dụng là một **Evidence Graph** gồm:

```
Documents
    ↓
Evidence
    ↓
Financial Facts
    ↓
Financial Metrics
    ↓
Risk Signals
    ↓
Credit Analysis
```

 Điều này giúp FinRisk AI đồng thời giải quyết ba vấn đề:

 1. **Document Intelligence** — hiểu hồ sơ tín dụng phi cấu trúc.
2. **Financial Intelligence** — trích xuất và tính toán financial facts/metrics chính xác.
3. **Credit Risk Intelligence** — liên kết các bằng chứng từ nhiều nguồn để phát hiện risk và hỗ trợ chuyên viên thẩm định.

 Hệ thống cuối cùng không trả lời đơn giản:

 > "Doanh nghiệp này rủi ro cao hay thấp?"

 mà phải trả lời được:

 > **"Rủi ro nào tồn tại, bằng chứng nào cho thấy điều đó, số liệu được lấy từ đâu, metric được tính như thế nào, các nguồn có mâu thuẫn không, và chuyên viên cần kiểm tra điều gì tiếp theo?"**

 Tôi đã cố ý đổi **trọng tâm của proposal** từ “Multi-Agent RAG” sang **Evidence → Fact → Metric → Risk**, vì đây mới là abstraction phù hợp với toàn bộ workflow thẩm định mà bạn vừa xác định.

 Một thay đổi kiến trúc đặc biệt quan trọng là: **Qdrant không còn là trung tâm của hệ thống**. Nó chỉ là một thành phần trong Retrieval Layer; `Financial Fact Store + Provenance + deterministic Formula Engine` mới là phần cốt lõi cho các bài toán như Altman Z, DSCR, D/E, liquidity và cross-document verification.

 Nếu bạn dùng proposal này để làm **capstone + research paper**, tôi cũng khuyên đổi tên Phase 2 từ **“Deep RAG cho Tài chính”** thành **“Financial Document Intelligence & Evidence-Aware Retrieval”**. Điều đó phản ánh đúng đóng góp kỹ thuật hơn và mở đường cho phần experimental comparison giữa fixed chunking, parent-child, table-preserving và evidence-aware retrieval.