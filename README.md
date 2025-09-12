<h1 style="color:blue;"><b>DS RPC 2.0 : ADEGuard – AI-Powered Adverse Drug Event Detection and Severity Mapping</b></h1>

This is the starter repository for Codebasics's Resume Project Challenge 2.0.  

This project focuses on building an AI-powered pipeline to detect **Adverse Drug Events (ADEs)** from symptom text, group them into symptom-based and age-specific clusters, and classify each event by severity.

Please **fork this repository** to get started.

---
 
## 📂 Data Access Instructions

Contestants will use the **VAERS** dataset provided by the U.S. Vaccine Adverse Event Reporting System.

1. Visit the official VAERS Data page:  
   👉 [https://vaers.hhs.gov/data/datasets.html](https://vaers.hhs.gov/data/datasets.html)

2. Scroll to the table listing data by year.

3. **Download the ZIP file** for your target year(s) from the **"Zip File"** column.  
   - Example: For 2025, click the link in the **Zip File** column (e.g., `4.95 MB`).
   - The ZIP will contain **three CSV files**:  
     - `VAERSDATA.csv` → Main case and patient data  
     - `VAERSSYMPTOMS.csv` → Coded adverse event terms using the **MedDRA** (Medical Dictionary for Regulatory Activities) terminology.  
         - Each report can have up to five coded symptoms (`SYMPTOM1`–`SYMPTOM5`), representing standardized MedDRA Preferred Terms.  
     - `VAERSVAX.csv` → Vaccine/product details

4. Extract the ZIP files for all target years, and move all three CSV files from each ZIP into the `data/raw` folder of this repository.

---

## 📝 Annotation Guidelines

Before starting annotation or model training, review the **Annotation Guidelines** in the `docs/` folder.  
They explain in detail:

- **ADE annotation rules** – how to identify Adverse Drug Events in text, including what to include and what to skip.
- **DRUG annotation rules** – how to label vaccine or drug mentions exactly as reported, handle brand names, code names, and generic terms.
- **Special cases** – rules for compound symptoms, repeated mentions, death/hospitalization references, and COVID-19 mentions.
- **Span formatting** – keeping longest medically accurate terms, excluding durations, and labeling each occurrence separately.
- **Quick checklist** – a step-by-step reminder to ensure annotations are consistent and compliant.

📌 **Tip:** Following these rules strictly ensures the labels are high quality and consistent, which is critical for training the NER model effectively.

---

### 📌 Learn More
Visit the **[challenge page](https://codebasics.io/challenges/codebasics-gen-ai-data-science-resume-project-challenge/22)** to learn more: **DS RPC-2.0**

My ADEGuard project started by AiswaryaG on Sep 11, 2025.
