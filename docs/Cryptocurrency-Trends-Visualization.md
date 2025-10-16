# Global Cryptocurrency Trends Visualization

**Author(s):** Francisco Chan, Valeria Ramírez, Esther Apaza, Carlos Helguera, Daniel Valdés, Rogelio Novelo  
**Date:** October 13th, 2025  
**Course:** Visual Modeling for Information  
**Program:** Data Engineering  
**Institution:** Universidad Politécnica de Yucatán  

---

## AI Assistance Disclosure

**AI Tool Used:** ChatGPT (GPT-5)  
**Type of Assistance:** Documentation writing, explanation structuring, and visualization guidance.  
**Extent of Use:** Moderate (~35% of total project workload).  
**Human Contribution:** Dataset collection and cleaning, Python implementation of visualizations, and validation of AI-generated content.  

> **Academic Integrity Statement:**  
> All AI-generated content was verified and understood by the authors. The team assumes full responsibility for its accuracy and integrity.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Objectives](#objectives)  
3. [Methodology](#methodology)  
4. [Data Sources](#data-sources)  
5. [Tools and Libraries](#tools-and-libraries)  
6. [Approach](#approach)  
7. [Implementation](#implementation)  
8. [Results](#results)  
9. [Conclusions](#conclusions)  
10. [References](#references)

---

## Project Overview

This project explores global search interest trends in cryptocurrencies using data from **Google Trends** between 2020 and 2025.  
The analysis focuses on how interest varies geographically and temporally for major cryptocurrencies like **Bitcoin**, **Ethereum**, and **Dogecoin**.

Using **Python**, **Plotly**, and **GeoPandas**, several interactive visualizations were created:

- **Global Choropleth Map** showing average interest per country.  
- **Animated Map (Time Slider)** to visualize changes over time.  
- **Regional Time Series** comparing trends across countries.  
- **Bar Chart** of the Top 10 countries by average interest.

These visualizations allow dynamic exploration of global patterns and reveal how geography and major crypto events influenced public attention.

---

## Objectives

- Visualize global and temporal variations in cryptocurrency interest.  
- Identify countries with the highest average search activity.  
- Compare multiple cryptocurrencies across different regions.  
- Build interactive visualizations using Plotly.  
- Understand geographic and temporal correlations of crypto popularity.

---

## Methodology

### Data Sources

- **Dataset:** `trends.csv`  
- **Origin:** Extracted via Google Trends.  
- **Fields:**
  - `date`: Weekly measurement date.  
  - `country`: Two-letter ISO country code.  
  - `keyword`: Cryptocurrency name (e.g., Bitcoin, Ethereum, Dogecoin).  
  - `interest`: Search index (0–100).  
- **Period Covered:** 2020–2025  

---

### Tools and Libraries

| Category | Tools |
|-----------|--------|
| Programming Language | Python |
| Data Handling | pandas, numpy |
| Geospatial Analysis | geopandas |
| Visualization | plotly.express, plotly.graph_objects, seaborn, matplotlib |
| Data Source | Google Trends |
| Mapping Data | Natural Earth shapefiles |

---

## Approach

### 1. Data Preparation
- Loaded and parsed `trends.csv`.  
- Normalized country codes and formatted dates.  
- Merged with geographic shapefiles for mapping.  

### 2. Geospatial Visualization
- Built a **choropleth map** showing global average interest per country.  
- Implemented an **animated slider** for temporal evolution (month by month).

### 3. Temporal Analysis
- Created **interactive line charts** with dropdowns for region/country selection.  
- Compared the evolution of interest across cryptocurrencies and continents.

### 4. Country Ranking
- Computed mean interest per country.  
- Displayed the **Top 10 countries** with flags for better visual identification.

---

## Implementation

### Phase 1: Data Cleaning and Integration
Standardized ISO country codes, parsed weekly date formats, and ensured numerical consistency.

### Phase 2: Geospatial Mapping
Constructed an interactive **choropleth map** using Plotly Express with:
- Hover tooltips (country name, average interest).
- Continuous color scale.
- Time slider for animation by date.

### Phase 3: Temporal Visualization
Developed **interactive time-series graphs** comparing Bitcoin, Ethereum, and Dogecoin interest levels per region.

### Phase 4: Ranking
Used Seaborn and Matplotlib to plot a **Top 10 countries bar chart**, enhanced with national flags and average scores.

---

## Results

### Key Visual Insights

1. **Global Leaders:**  
   Japan, South Korea, Brazil, Germany, and the United States emerged as the countries with the highest average interest in cryptocurrencies — with **Japan leading** overall.

2. **Most Popular Cryptocurrencies:**  
   **Bitcoin** and **Ethereum** consistently ranked as the most searched and discussed across nearly all regions, dominating public attention globally.

3. **Temporal Peaks:**  
   During **early 2021**, global search interest saw a significant rise, coinciding with the massive price surge and crypto adoption wave.  
   In contrast, by **October 2022**, global attention sharply declined as market activity cooled.

4. **Regional Patterns:**  
   Asian countries, particularly **Japan** and **China**, currently show the highest sustained interest in cryptocurrency topics.  
   Europe and the Americas also display steady attention, though with more variability across months.

5. **Dogecoin Trends:**  
   A distinct **spike in May 2021** occurred globally for Dogecoin, following viral online campaigns and celebrity endorsements.  
   However, this peak **faded quickly** within a few months, reflecting the coin’s short-term speculative popularity.

---

### Visual Outputs

| Visualization Type | Description |
|--------------------|--------------|
| **Global Map** | Shows the average crypto interest by country. |
| **Animated Map (Slider)** | Displays monthly evolution of global interest. |
| **Regional Trend Lines** | Compare how different regions evolved over time. |
| **Top 10 Bar Chart** | Highlights countries with the highest mean interest. |

---

### Performance Summary

| Metric | Value | Description |
|--------|--------|-------------|
| Dataset Size | ~2000 rows | Weekly global data (2020–2025) |
| Visualization Tools | Plotly, Seaborn | Interactive and static visuals |
| Average Processing Time | < 3 seconds | Lightweight and responsive |
| AI Utilization | ~35% | Focused on documentation and code explanations |

---

## Conclusions

### Summary

This project successfully integrated **geospatial and temporal visualizations** to analyze worldwide patterns of public interest in cryptocurrencies.  
By combining data from **Google Trends** with Python-based analytical tools such as **Plotly** and **GeoPandas**, the team created a set of interactive graphics that reveal how attention to digital assets evolved across countries and time.

The results clearly demonstrate that **interest in cryptocurrencies is both regionally and temporally dynamic**.  
Asian countries — particularly **Japan and South Korea** — consistently appear as leading centers of public engagement, reflecting their strong technological infrastructure and early adoption of digital finance.  
Meanwhile, nations like **Brazil**, **Germany**, and the **United States** also display high levels of search activity, highlighting the global nature of cryptocurrency awareness.

Temporal analysis revealed distinct waves of attention linked to real-world events.  
The **first half of 2021** marked a global surge in search interest, corresponding to rapid price growth and widespread media coverage, while **late 2022** showed a notable decline in global curiosity due to market downturns and reduced social media influence.  
Additionally, the **Dogecoin phenomenon** in May 2021 represented a unique case of social-driven speculation, with a sharp spike in searches followed by a rapid fall-off, illustrating how online communities can temporarily reshape public attention.

Overall, the project demonstrates how **visual modeling and temporal mapping** can transform large-scale search data into meaningful insights about collective behavior, economic sentiment, and cultural trends.  
It highlights the potential of open data and visualization tools to understand **global digital phenomena** and track emerging interests over time with precision and clarity.


### Lessons Learned
- Normalizing ISO codes is crucial for accurate joins in global datasets.  
- Animated visualizations improve trend interpretation dramatically.  
- Small or missing countries in GeoJSON files can create data gaps.  
- Interactivity significantly enhances user understanding and engagement.

### Future Work
- Automate data extraction from the Google Trends API.  
- Expand analysis to include **newer cryptocurrencies** (e.g., Solana, Ripple).  
- Apply **trend smoothing** for long-term comparisons.  
- Integrate final visualizations into a **Plotly Dash** or **Streamlit** web app.

---

## References

- [Google Trends API (Pytrends)](https://pypi.org/project/pytrends/)  
- [Natural Earth Dataset](https://www.naturalearthdata.com/)  
- [Plotly Express Documentation](https://plotly.com/python/plotly-express/)  
- [GeoPandas Documentation](https://geopandas.org/)  
- [Seaborn Documentation](https://seaborn.pydata.org/)

---
