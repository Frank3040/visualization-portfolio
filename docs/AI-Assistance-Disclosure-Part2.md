# AI Assistance Disclosure — Cryptocurrency Dashboard & Data Analysis Project  

- **AI Tool Used:** ChatGPT (GPT-5)  
- **Overall Assistance Level:** 60%  
- **Main Use Cases**
    - Code generation:** 70%  
    - Documentation:** 45%  
    - Debugging:** 55%  
    - Data analysis & cleaning:** 65%  
    - Visualizations (Seaborn / Plotly):** 60%  
- **Human Contributions**
    - Conceptual design of dashboard sections (map, time series, correlations)  
    - Integration of datasets (`sample_temporal.csv` and `trends.csv`)  
    - Validation of merged data and cleaning procedures  
    - Adjustments to Seaborn & Plotly layout (titles, colors, filters)  
    - Original interpretation and written analysis for each visualization  

- **Verification Process**
    - Manual review of all AI-generated Python and visualization scripts  
    - Step-by-step validation in Jupyter Notebook / Colab  
    - Comparison between expected and generated time-series & map results  
    - Testing of merge operations between trends and temporal datasets  


## **Work Session Log — Prompts and AI Use**

| Session | Duration | Goals | AI Tool | AI Time | Prompt | Type of Assistance | AI Output | Human Modifications | AI Estimate |
|----------|-----------|--------|----------|----------|---------|------------------|------------------|------------------|--------------|
| 1 | 2 h | Create interactive world map of crypto interest | ChatGPT | 1 h | “Generate a choropleth map showing crypto interest by country using Plotly Express” | Visualization | Working map with dropdown filters | Adjusted labels, color scale & layout | 55% |
| 2 | 1.5 h | Clean and structure sample_temporal.csv (multi-row header) | ChatGPT | 40 min | “Fix multi-index CSV and rename columns for ETH, BTC, SOL, DOGE” | Data cleaning | Script to flatten headers and rename columns | Minor changes to keep correct column names | 50% |
| 3 | 1 h | Merge trends.csv with temporal data | ChatGPT | 30 min | “Merge interest-by-country dataset with temporal crypto data by date” | Data wrangling | Merge and conversion script | Adjusted date parsing and join type | 45% |
| 4 | 1.5 h | Generate Time Series by Region visualization | ChatGPT | 45 min | “Create Seaborn time series plot grouped by region” | Visualization | Lineplot by country (hue) | Added title, labels, and filters | 50% |
| 5 | 1 h | Debug KeyError issues ('Date', 'date') | ChatGPT | 30 min | “Fix KeyError when reading multi-row CSV header and date parsing” | Debugging | Fixed column reference logic | Adjusted rename strategy | 55% |
| 6 | 1 h | Design final dashboard structure (map + time series) | ChatGPT | 30 min | “Combine map and time-series plots into unified dashboard” | Code generation | Layout and filter structure | Modified responsive layout | 50% |
| 7 | 1 h | Document AI involvement and analysis process | ChatGPT | 30 min | “Write AI assistance disclosure for crypto dashboard project” | Documentation | Draft of AI disclosure | Refined wording and structure | 55% |


**AI Assistance Calculation (Simulated)**

1. **Time %:**  
   Total project time = **10 h**, AI active time ≈ **5 h**  
   → (5 / 10) × 100 = **50%**

2. **Content %:**  
   Code + visualizations + documentation ≈ **60%** of total project content  

3. **Complexity %:**  
   Data integration + visualization logic → **Level 3 (Intermediate)** → 62.5%  

4. **Self-Assessment Score:**  
   - Q1 (Conceptualization): 55%  
   - Q2 (Implementation): 65%  
   - Q3 (Understanding): 60%  
   - Q4 (Problem-Solving): 55%  
   → **Average:** (55 + 65 + 60 + 55)/4 = **58.75%**

 ✅ **Final AI Assistance % Calculation**
0.25×50 + 0.35×60 + 0.25×62.5 + 0.15×58.75 ≈ **59.1%**

→ Final Reported AI Assistance Level: 60%**